#if 0
/opt/rocm/llvm/bin/clang++ \
    -std=c++20 \
    -I${ROCM_PATH}/include \
    -D__HIP_PLATFORM_AMD__ \
    -fPIC -shared -g -O3 \
    -o libpreload-me.so preload-me.cpp
exit 0
#endif

#include <stdio.h>
#include <dlfcn.h>
#include <pthread.h>
#include <map>
#include <vector>
#include <cassert>
#include <cstring>

// #include <hip/hip_runtime.h>

#define hipError_t int
#define hipSuccess 0
#define hipEvent_t void*
#define hipStream_t void*
#define hipMemAttachGlobal 0x01
#define hipErrorNotSupported 801

extern "C" {
    hipError_t hipDeviceSynchronize	(void);
    hipError_t hipStreamSynchronize (hipStream_t stream);
    
    void roctracer_start();
    void roctracer_stop();
}

namespace {

// Return the ID of the current thread
// inline pthread_id_np_t tid() {
//     return pthread_getthreadid_np();
// }
inline pthread_t tid() {
    return pthread_self();
}

// Initializes the symbol of the original runtime symbol and return 0 if success
template<typename T>
int lazy_init(T *&fptr, const char *name) {
    void *&ptr = reinterpret_cast<void *&>(fptr);

    if (ptr) return 0;

    ptr = dlsym(RTLD_NEXT, name);

    assert(ptr);

    return ptr ? 0 : -1;
}

// hipError_t hipFree (void *ptr)
hipError_t (*hipFree_orig)(void *) = nullptr;
// hipError_t 	hipFreeAsync (void *dev_ptr, hipStream_t stream)
hipError_t (*hipFreeAsync_orig)(void *, hipStream_t) = nullptr;
// hipError_t 	hipMalloc (void **ptr, size_t size)
hipError_t (*hipMalloc_orig)(void **, size_t) = nullptr;
// hipError_t 	hipMallocAsync (void **dev_ptr, size_t size, hipStream_t stream)
hipError_t (*hipMallocAsync_orig)(void **, size_t, hipStream_t) = nullptr;
// hipError_t 	hipMallocManaged (void **dev_ptr, size_t size, unsigned int flags __dparm(hipMemAttachGlobal))
hipError_t (*hipMallocManaged_orig)(void **, size_t, unsigned int) = nullptr;
// hipError_t 	hipMemPrefetchAsync (const void *dev_ptr, size_t count, int device, hipStream_t stream __dparm(0))
hipError_t (*hipMemPrefetchAsync_orig)(const void *, size_t, int, hipStream_t) = nullptr;

// Name of the range of interest.
const char *ProfileRangeName = "Propagate";
// Instance we want of the captured region.
int ProfileInstance = 3;
// Variable to track the level of the captured region.
int ProfileLevel = -1;
// Thread that matters
pthread_t ProfileThread = 0;

// int roctxRangePushA(const char* message)
int (*roctxRangePushA_orig)(const char*) = nullptr;
// int roctxRangePop()
int (*roctxRangePop_orig)() = nullptr;

pthread_mutex_t MapMutex = PTHREAD_MUTEX_INITIALIZER;

class ThreadData {
    bool IsInitialized;

    // Map to hlp account for allocations made.
    typedef std::map<size_t,size_t> SizeToCountsMapType;
    SizeToCountsMapType SizeMap;

    // Number of increments done due to allocations in this thread.
    size_t TotalIncrements;

    // Static sizes considered. Data below needs to be consistent.
    const size_t SizesI = 64;
    const size_t SizesE = 262144;

#define NNN 32ul
    // Slot counts of a given size - minimum is 64 bytes.
    const size_t Counts[13] = {
        NNN * 262144,  // 64
        NNN * 4096,    // 128
        NNN * 4096,    // 256
        NNN * 4096,    // 512
        NNN * 4096,    // 1024
        NNN * 2048,     // 2048
        NNN * 1024,     // 4096
        NNN * 1024,     // 8192
        NNN * 1024,     // 16384
        NNN * 512,     // 32768
        NNN * 512,     // 65536
        NNN * 512,     // 131072
        NNN * 512      // 262144
    };

    void *MemoryChunk;
    size_t memoryChunkSize;

    char *chunk() {
        return reinterpret_cast<char*>(MemoryChunk);
    }
    
    // A set bit in this matrix means that a slot is allocated.
    std::vector<std::vector<uint64_t>> Register;

public:
    ThreadData() : IsInitialized(false), TotalIncrements(0), MemoryChunk(nullptr), memoryChunkSize(0)  {}
    void init() {
        if (IsInitialized)
            return;

        lazy_init(hipMallocManaged_orig, "hipMallocManaged");

        for(size_t i=0, c=SizesI; c <= SizesE; ++i, c*=2) {
            memoryChunkSize += c * Counts[i];

            // Each entry position controls 64 bytes
            Register.emplace_back(Counts[i]/64,0);
        }

        if(hipMallocManaged_orig(&MemoryChunk, memoryChunkSize, hipMemAttachGlobal))
            printf("[thread %ld] -> failed to allocate %ld bytes.\n", tid(), memoryChunkSize);
        else
            printf("[thread %ld] -> allocated %ld bytes [%p-%p[ .\n", tid(), memoryChunkSize, MemoryChunk, reinterpret_cast<char*>(MemoryChunk) + memoryChunkSize);

        IsInitialized = true;
    }

    ~ThreadData() {

        if(!MemoryChunk)
            return;

        lazy_init(hipFree_orig, "hipFree");
        if(hipFree_orig(MemoryChunk))
            printf("[thread %ld] -> failed to deallocate its chunk.\n", tid());
        else
            printf("[thread %ld] -> deallocated its chunk.\n", tid());
    }

    // Returns a pointer to a slot in the allocated chunk or null for normal treatment.
    void *allocate(size_t size) {
        init();

        // - Find the register entry for a given size.
        // - Set the bit for the relevant entry.
        // - Calculate the offset in the chunk for it.
        // - If size too big or not enough slots, return null.
        for(size_t i=0, c=SizesI, offset=0; c <= SizesE; offset +=c*Counts[i], ++i, c*=2) {

            if ( size > c)
                continue;

            auto &SizeReg = Register[i];

            pthread_mutex_lock(&MapMutex);
            for(size_t RegElem = 0; RegElem < SizeReg.size(); ++RegElem )
                if (uint64_t entry = SizeReg[RegElem]; entry != -1ul) {
                    size_t RegBit = 0;
                    uint64_t mask = 0x01;
                    for(; entry & mask; ++RegBit, mask <<= 1);
                    SizeReg[RegElem] ^= mask;
                    pthread_mutex_unlock(&MapMutex);

                    char *ret = chunk() + offset + (RegElem*64 + RegBit) * c;

                    // printf("[thread %ld] -> allocated size %ld [elem %ld - bit %ld] - %p(%ld).\n", tid(), c, RegElem, RegBit,ret,size);

                    return ret;
                }

            pthread_mutex_unlock(&MapMutex);

            // Not enough slots for the size.
            printf("[thread %ld] -> not enough slots of size %ld.\n", tid(), c);

            return nullptr;
        }
        return nullptr;
    }

    // Return zero if the data has been successfully deallocated.
    int free(void *ptr, hipStream_t stream = 0) {
        init();

        char *p = reinterpret_cast<char *>(ptr);

        // printf("[thread %ld] -> attempting to free %p .\n", tid(), ptr);
        
        // check if this pointer falls in the range controlled.
        if(p < chunk() || p >= chunk() + memoryChunkSize)
            return -1;

        for(size_t i=0, c=SizesI, offset=0; c <= SizesE; offset +=c*Counts[i], ++i, c*=2) {

            if ( p >= chunk() + offset + c*Counts[i] )
                continue;

            auto &SizeReg = Register[i];

            size_t Index = (p - (chunk() + offset))/c;
            size_t RegElem = Index / 64;
            size_t RegBit = Index % 64;

            if(stream) {
                if (hipStreamSynchronize(stream)) {printf("Stream Synchronize issue in memory pool!!!\n");}
            } else {
                if (hipDeviceSynchronize()) {printf("Device Synchronize issue in memory pool!!!\n");}
            }

            pthread_mutex_lock(&MapMutex);
            SizeReg[RegElem] ^= 0x01ul << RegBit;
            pthread_mutex_unlock(&MapMutex);

            //printf("[thread %ld] -> freed size %ld [elem %ld - bit %ld] - %p .\n", tid(), c, RegElem, RegBit, ptr);

            return 0;
        }
        return -1;
    }

    void incSize(size_t s) { 
        SizeMap.try_emplace(s,0);
        ++SizeMap[s];

        // if(++TotalIncrements % 128 == 0)
        //     printInfo();
    }
    void decSize(size_t s) {
        --SizeMap[s];
    }
    void printInfo() {
        for(auto II=SizeMap.begin(), IE=SizeMap.end(); II != IE; ++II) {
            printf("[thread %ld] -> size %ld -> %ld counts.\n", tid(), II->first, II->second);
        }
    }
};

ThreadData wm;
ThreadData &getData() {
    pthread_t t = tid();
    return wm;
}

} // namespace

extern "C" {
hipError_t 	hipFree (void *ptr) {
    if(lazy_init(hipFree_orig, "hipFree")) return hipErrorNotSupported;

    //printf("In sfantao implementation of hipFree\n");
    if (!getData().free(ptr))
        return hipSuccess;

    return hipFree_orig(ptr);
}
hipError_t 	hipFreeAsync (void *ptr, hipStream_t stream){
    if(lazy_init(hipFreeAsync_orig, "hipFreeAsync")) return hipErrorNotSupported;

    //printf("In sfantao implementation of hipFreeAsync\n");
    if (!getData().free(ptr, stream))
        return hipSuccess;
    return hipFreeAsync_orig(ptr, stream);
}
hipError_t 	hipMalloc (void **ptr, size_t size){
    if(lazy_init(hipMalloc_orig, "hipMalloc")) return hipErrorNotSupported;

    //printf("In sfantao implementation of hipMalloc\n");
    //if (*ptr = getData().allocate(size) ; *ptr)
    //    return hipSuccess;

    return hipMalloc_orig(ptr, size);
}
hipError_t 	hipMallocAsync (void **ptr, size_t size, hipStream_t stream){
    if(lazy_init(hipMallocAsync_orig, "hipMallocAsync")) return hipErrorNotSupported;

    //printf("In sfantao implementation of hipMallocAsync\n");
    //if (*ptr = getData().allocate(size) ; *ptr)
    //    return hipSuccess;
    return hipMallocAsync_orig(ptr, size, stream);
}
hipError_t 	hipMallocManaged (void **ptr, size_t size, unsigned int flags = hipMemAttachGlobal){
    if(lazy_init(hipMallocManaged_orig, "hipMallocManaged")) return hipErrorNotSupported;

    //printf("In sfantao implementation of hipMallocManaged\n");
    if (*ptr = getData().allocate(size) ; *ptr)
        return hipSuccess;
    return hipMallocManaged_orig(ptr, size, flags);
}
hipError_t 	hipMemPrefetchAsync (const void *dev_ptr, size_t count, int device, hipStream_t stream = 0){
    // Disable
    return hipSuccess;

    // This is known to not be better than first touch;
    // return hipSuccess;

    if(lazy_init(hipMemPrefetchAsync_orig, "hipMemPrefetchAsync")) return hipErrorNotSupported;

    //printf("In sfantao implementation of hipMemPrefetchAsync\n");
    return hipMemPrefetchAsync_orig(dev_ptr, count, device, stream);
}

int roctxRangePushA(const char* message) {

    if(lazy_init(roctxRangePushA_orig, "roctxRangePushA")) return -3;

    // Do we still need to look for our region?
    if (ProfileInstance > 0)
        // Does the region have the targe name?
        if(!strcmp(ProfileRangeName, message)) {
            --ProfileInstance;
            // Is this the right instance?
            if(!ProfileInstance) {

                // Start profile!
                roctracer_start();

                ProfileThread = tid();
                ProfileLevel = 0;

                printf(" ----> Starting profile for region %s - [%ld].\n", message, ProfileThread);

                return roctxRangePushA_orig(message);
            }
        }

    if (ProfileThread && ProfileThread == tid() && ProfileLevel >= 0)
        ++ProfileLevel;

    return roctxRangePushA_orig(message);
}

int roctxRangePop() {

    if(lazy_init(roctxRangePop_orig, "roctxRangePop")) return -3;

    int ret = roctxRangePop_orig();

    // Only th allowed thread can touch this.
    if(ProfileThread && ProfileThread == tid()) {
        if (ProfileLevel == 0) {
            printf(" ----> Stopped profile for region %d.\n", ProfileLevel);
            roctracer_stop();
            ProfileLevel = -1;
        } else if (ProfileLevel > 0)
            --ProfileLevel;
    }

    return ret;
}


}
