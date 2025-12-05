# gelu

Ok so I wanted to get good with basic GPU optimizations like global memory coalescing, shared memory caching etc but looks like gelu was not a good algorithm to do that.

I implemented a baseline version and realized I was memory bound (expected) but then my coalescing was already optimized due to the nature of the kernel and because every elem is only used once shared memory caching also is not something that can be implemented.

Going to create a new repo that is more interesting by maybe trying to do something revolving around a stencil operation

_________________
Outputs and Memory bandwidth calculations:
_________________

(main) root@C.28507991:/workspace/gelu/build$ ./bin/gelu 
--- GELU Test ---
Array Size N: 16777216 (0.0625 GB)
Launching kernel with 65536 blocks and 256 threads/block.
Kernel execution time (rough): 386.816 us
Verification Check: PASSED (midpoint value: 0.841345)
(main) root@C.28507991:/workspace/gelu/build$ 


main) root@C.28507991:/workspace/gelu/build$ ./bin/gelu 
--- GELU Stable Timing Test ---
Array Size N: 16777216 (0.0625 GB)
Grid: 65536 blocks, 256 threads/block.
Warming up the GPU and Caches (10 runs)...
Starting Stable Timing Loop (100 runs)...

--- Timing Results ---
Total execution time for 100 stable runs: 11.562 ms
**Average kernel execution time:** 115.62 us -> 

Verification Check: **PASSED** (midpoint value: 0.841345)



total amount of data moved (read + written)
read and write a float - 4 bytes + 4 bytes = 8 bytes
8 bytes * N (16,777,216) = 134,217,728 bytes

134,217,728 bytes / 0.00011562 seconds = 1.1608521709 TB/s

The A100 40GB model offers up to 1,555 GB/s (gigabytes per second) of memory bandwidth. aka 1.555 TB/s

But i'm getting 1.16 TB/s aka almost 75% of that memory bandwidth

That tells me that i'm memory bound