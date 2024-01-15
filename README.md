## Memory pool for Vlasiator porting tests.

This library implements a simple slot-based memory pool. The slots are controlled by regulating:
```
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
```
This means that the pool will maintain `NNN * 262144` for 1 to 64 byte allocations, `NNN * 4096` for 65 to 128 byte allocations and so on and so forth. If the implementaions runs out of slots it prints a message.

When initialized the implementation allocates a single piece of HIP managed memory and provides pointer to slots there as needed.

It is expected to be thread safe.

It can be extended to support other allocators beyond managed memory.

### Run
To run should be sufficient to do: `LD_PRELOAD=<path to library>/libpreload-me.so ./vlasiator ...`. 

### Build

The library can be built with: `bash preload-me.cpp`. 
