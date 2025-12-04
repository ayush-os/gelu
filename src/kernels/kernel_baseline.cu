#include <math.h>

// --- Baseline Kernel (Naive, Unoptimized) ---
// This uses a direct 1-to-1 mapping from thread index to array index.

__global__ void baseline_elementwise_kernel(float* output, const float* input, int N) {
    const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId >= N) return;

    float in = input[threadId];

    output[threadId] = in * __nv_normcdf(in);
}