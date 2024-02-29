#if 0
/opt/rocm/llvm/bin/clang++ \
    -std=c++20 \
    -I${ROCM_PATH}/include \
    -D__HIP_PLATFORM_AMD__ \
    -fPIC -shared -g -O0 \
    -o libpreload-me.so preload-me.cpp
exit $?
#endif

#include <stdio.h>
#include <dlfcn.h>
#include <pthread.h>
#include <map>
#include <vector>
#include <cassert>
#include <cstring>
#include <bitset>

#define FILTER_SYNCS

// #include <hip/hip_runtime.h>

#define hipError_t int
#define hipSuccess 0
#define hipEvent_t void*
#define hipStream_t void*
#define hipMemPool_t void*
#define hipMemAttachGlobal 0x01
#define hipErrorNotSupported 801
enum hipMemPoolAttr {
    hipMemPoolAttrDummy = 0x0
};
enum hipMemcpyKind {
  hipMemcpyHostToHost = 0,
  hipMemcpyHostToDevice = 1,
  hipMemcpyDeviceToHost = 2,
  hipMemcpyDeviceToDevice = 3,
  hipMemcpyDefault
};
struct dim3 {
    uint32_t x;
    uint32_t y;
    uint32_t z;
};

extern "C" {
    hipError_t hipMemGetInfo(size_t *, size_t *);
    hipError_t hipStreamSynchronize(hipStream_t stream);
    hipError_t hipDeviceSynchronize();
    void roctracer_start();
    void roctracer_stop();
}

namespace {

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
// hipError_t hipStreamSynchronize(hipStream_t stream)
hipError_t (*hipStreamSynchronize_orig)(hipStream_t) = nullptr;
// hipError_t hipDeviceSynchronize()
hipError_t (*hipDeviceSynchronize_orig)() = nullptr;
// hipError_t hipLaunchKernel(const void* function_address, dim3 numBlocks, dim3 dimBlocks, void** args, size_t sharedMemBytes, hipStream_t stream)
hipError_t (*hipLaunchKernel_orig)(const void*, dim3, dim3, void**, size_t, hipStream_t) = nullptr;
// hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind copyKind, hipStream_t stream)
hipError_t (*hipMemcpyAsync_orig)(void*, const void*, size_t, hipMemcpyKind, hipStream_t) = nullptr;
// hipError_t hipMemsetAsync(void* devPtr, int value, size_t count, hipStream_t stream)
hipError_t (*hipMemsetAsync_orig)(void*, int, size_t, hipStream_t) = nullptr;
// hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind copyKind)
hipError_t (*hipMemcpy_orig)(void*, const void*, size_t, hipMemcpyKind) = nullptr;
// hipError_t hipDeviceGetDefaultMemPool(hipMemPool_t* mem_pool, int device)
hipError_t (*hipDeviceGetDefaultMemPool_orig)(hipMemPool_t*, int) = nullptr;
// hipError_t hipMemPoolGetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void* value)
hipError_t (*hipMemPoolGetAttribute_orig)(hipMemPool_t mem_pool, hipMemPoolAttr attr, void* value) = nullptr;
// hipError_t hipMemPoolSetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void* value)
hipError_t (*hipMemPoolSetAttribute_orig)(hipMemPool_t mem_pool, hipMemPoolAttr attr, void* value) = nullptr;

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
pthread_mutex_t ActivityMutex = PTHREAD_MUTEX_INITIALIZER;

template<bool IsManaged>
class PoolData {
    bool IsInitialized;

    // // Map to hlp account for allocations made.
    // typedef std::map<size_t,size_t> SizeToCountsMapType;
    // SizeToCountsMapType SizeMap;

    // // Number of increments done due to allocations in this thread.
    // size_t TotalIncrements;

    // Static sizes considered. Data below needs to be consistent.
    const size_t SizesI = 64;
    const size_t SizesE = 262144;

#define NNN 64ul
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
    PoolData() : IsInitialized(false), MemoryChunk(nullptr), memoryChunkSize(0)  {}
    void init() {
        if (IsInitialized)
            return;

        if (IsManaged)
            lazy_init(hipMallocManaged_orig, "hipMallocManaged");
        else
            lazy_init(hipMalloc_orig, "hipMalloc");

        for(size_t i=0, c=SizesI; c <= SizesE; ++i, c*=2) {
            memoryChunkSize += c * Counts[i];

            // Each entry position controls 64 bytes
            Register.emplace_back(Counts[i]/64,0);
        }

        hipError_t res = hipSuccess;
        if(IsManaged)
            res = hipMallocManaged_orig(&MemoryChunk, memoryChunkSize, hipMemAttachGlobal);
        else
            res = hipMalloc_orig(&MemoryChunk, memoryChunkSize);

        if(res)
            printf("[thread %ld] -> failed to allocate %ld bytes (%s).\n", tid(), memoryChunkSize, IsManaged ? "managed":"regular");
        else
            printf("[thread %ld] -> allocated %ld bytes [%p-%p[ (%s).\n", tid(), memoryChunkSize, MemoryChunk, reinterpret_cast<char*>(MemoryChunk) + memoryChunkSize, IsManaged ? "managed":"regular");

        IsInitialized = true;
    }

    ~PoolData() {

        if(!MemoryChunk)
            return;

        lazy_init(hipFree_orig, "hipFree");
        if(hipFree_orig(MemoryChunk))
            printf("[thread %ld] -> failed to deallocate %s chunk.\n", tid(), IsManaged ? "managed":"regular");
        else
            printf("[thread %ld] -> deallocated %s chunk.\n", tid(), IsManaged ? "managed":"regular");
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
            printf("[thread %ld] -> not enough slots of size %ld (%s).\n", tid(), c, IsManaged ? "managed":"regular");

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
};

// Managed pool and regular pool.
PoolData</*IsManaged=*/true> mpool;
PoolData</*IsManaged=*/false> rpool;

// Track activity per stream
enum ActivityType {
    at_none,
    at_sync,
    at_kernel,
    at_todev,
    at_fromdev,
    at_alloc,
    at_free,
};
typedef std::pair<ActivityType,hipStream_t> Action;

class ActivityTracker {
    std::map<pthread_t,Action> LastKnownActivity;
public:
    ActivityTracker() {}

    void set(ActivityType type, hipStream_t stream = 0) {
#ifndef FILTER_SYNCS
        return;
#endif
        auto v = LastKnownActivity.find(tid());
        if (v == LastKnownActivity.end()) {
            pthread_mutex_lock(&ActivityMutex);
            LastKnownActivity[tid()] = Action(type,stream);
            pthread_mutex_unlock(&ActivityMutex);
            return;
        }
        v->second = Action(type,stream);
    }
    Action& get() {

        auto v = LastKnownActivity.find(tid());

        if (v == LastKnownActivity.end()) {
            pthread_mutex_lock(&ActivityMutex);
            Action& ret = LastKnownActivity[tid()] = Action(at_none,0);
            pthread_mutex_unlock(&ActivityMutex);
            return ret;
        }
        return v->second;
    }
};

ActivityTracker tracker;

} // namespace

static size_t hipMallocCount = 0;

extern "C" {
hipError_t 	hipFree (void *ptr) {

    tracker.set(at_free);

    if(lazy_init(hipFree_orig, "hipFree")) return hipErrorNotSupported;

    if (!mpool.free(ptr))
        return hipSuccess;
    if (!rpool.free(ptr))
        return hipSuccess;

    return hipFree_orig(ptr);
}
hipError_t 	hipFreeAsync (void *ptr, hipStream_t stream){

    tracker.set(at_free,stream);

    if(lazy_init(hipFreeAsync_orig, "hipFreeAsync")) return hipErrorNotSupported;

    if (!mpool.free(ptr, stream))
        return hipSuccess;
    if (!rpool.free(ptr))
        return hipSuccess;
    return hipFreeAsync_orig(ptr, stream);
}

hipError_t 	hipMalloc (void **ptr, size_t size){

    tracker.set(at_alloc);

    if(lazy_init(hipMalloc_orig, "hipMalloc")) return hipErrorNotSupported;

    if (*ptr = rpool.allocate(size) ; *ptr)
        return hipSuccess;

    return hipMalloc_orig(ptr, size);
}
hipError_t 	hipMallocAsync (void **ptr, size_t size, hipStream_t stream){

    tracker.set(at_alloc, stream);

    if(lazy_init(hipMallocAsync_orig, "hipMallocAsync")) return hipErrorNotSupported;

    if (*ptr = rpool.allocate(size) ; *ptr)
        return hipSuccess;
    return hipMallocAsync_orig(ptr, size, stream);
}
hipError_t 	hipMallocManaged (void **ptr, size_t size, unsigned int flags = hipMemAttachGlobal){

    tracker.set(at_alloc);

    if(lazy_init(hipMallocManaged_orig, "hipMallocManaged")) return hipErrorNotSupported;

    if (*ptr = mpool.allocate(size) ; *ptr)
        return hipSuccess;
    return hipMallocManaged_orig(ptr, size, flags);
}
hipError_t 	hipMemPrefetchAsync (const void *dev_ptr, size_t count, int device, hipStream_t stream = 0){
    // Disable.
    return hipSuccess;

    if(lazy_init(hipMemPrefetchAsync_orig, "hipMemPrefetchAsync")) return hipErrorNotSupported;

    return hipMemPrefetchAsync_orig(dev_ptr, count, device, stream);
}

#ifdef FILTER_SYNCS
hipError_t hipStreamSynchronize(hipStream_t stream) {

    // Previous action.
    auto[paction, pstream] = tracker.get();

    // if (paction != at_free && paction != at_alloc) {
        // If the last known activity is synchronous don't need to synchronize again.
        if (!pstream)
            return hipSuccess;
        else {
            // We "likely" don't need to synchronize for transfers to the device and synchronizations on the same stream.
            if (paction == at_sync || paction == at_todev)
                return hipSuccess;
        }
    // }

    tracker.set(at_sync, stream);

    if(lazy_init(hipStreamSynchronize_orig, "hipStreamSynchronize")) return hipErrorNotSupported;

    return hipStreamSynchronize_orig(stream);
}

hipError_t hipDeviceSynchronize() {

    // Previous action.
    auto[paction, pstream] = tracker.get();

    // if (paction != at_free && paction != at_alloc) {
        // If the last known activity is synchronous don't need to synchronize again.
        if (!pstream)
            return hipSuccess;
    // }

    tracker.set(at_sync);

    if(lazy_init(hipDeviceSynchronize_orig, "hipDeviceSynchronize")) return hipErrorNotSupported;

    return hipDeviceSynchronize_orig();
}

hipError_t hipLaunchKernel(const void* function_address, dim3 numBlocks, dim3 dimBlocks, void** args, size_t sharedMemBytes, hipStream_t stream) {
    
    tracker.set(at_kernel, stream);
    
    if(lazy_init(hipLaunchKernel_orig, "hipLaunchKernel")) return hipErrorNotSupported;

    return hipLaunchKernel_orig(function_address,numBlocks,dimBlocks,args,sharedMemBytes,stream);
}
hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind copyKind, hipStream_t stream) {

    tracker.set((copyKind == hipMemcpyDeviceToHost)? at_fromdev : at_todev, stream);

    if(lazy_init(hipMemcpyAsync_orig, "hipMemcpyAsync")) return hipErrorNotSupported;

    return hipMemcpyAsync_orig(dst, src, sizeBytes, copyKind, stream);
}
hipError_t hipMemsetAsync(void* devPtr, int value, size_t count, hipStream_t stream) {

    tracker.set(at_todev, stream);

    if(lazy_init(hipMemsetAsync_orig, "hipMemsetAsync")) return hipErrorNotSupported;

    return hipMemsetAsync_orig(devPtr, value, count, stream);
}
hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind copyKind) {

    tracker.set((copyKind == hipMemcpyDeviceToHost)? at_fromdev : at_todev);

    if(lazy_init(hipMemcpy_orig, "hipMemcpy")) return hipErrorNotSupported;

    return hipMemcpy_orig(dst, src, sizeBytes, copyKind);
}
#endif

hipError_t hipDeviceGetDefaultMemPool(hipMemPool_t* mem_pool, int device) {
    // disable
    *mem_pool = nullptr;
    return hipSuccess;

    if(lazy_init(hipDeviceGetDefaultMemPool_orig, "hipDeviceGetDefaultMemPool")) return hipErrorNotSupported;

    return hipDeviceGetDefaultMemPool_orig(mem_pool, device);
}
hipError_t hipMemPoolGetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void* value) {
    // disable
    return hipSuccess;

    if(lazy_init(hipMemPoolGetAttribute_orig, "hipMemPoolGetAttribute")) return hipErrorNotSupported;

    return hipMemPoolGetAttribute_orig(mem_pool, attr, value);
}
hipError_t hipMemPoolSetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void* value) {
    // disable
    return hipSuccess;

    if(lazy_init(hipMemPoolSetAttribute_orig, "hipMemPoolSetAttribute")) return hipErrorNotSupported;

    return hipMemPoolSetAttribute_orig(mem_pool, attr, value);
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

    // Only the allowed thread can touch this.
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
