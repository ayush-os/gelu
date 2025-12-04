#include <math.h>

// --- Baseline Kernel (Naive, Unoptimized) ---
// This uses a direct 1-to-1 mapping from thread index to array index.

__global__ void baseline_elementwise_kernel(float* output, const float* input, int N) {
    const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId >= N) return;

    float in = input[threadId];

    float sqrt2_recip = 0.70710678118f; // 1.0f / sqrt(2.0f)
    float term = in * sqrt2_recip; 

    // GELU(x) = x * Phi(x)
    // Phi(x) = 0.5 * (1 + erff(x / sqrt(2)))
    output[threadId] = in * 0.5f * (1.0f + erff(term));
}