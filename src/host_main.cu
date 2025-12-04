#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

// Function declarations
__global__ void baseline_elementwise_kernel(float* output, const float* input, int N);

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int N_BITS = 24;
    const int N = 1 << N_BITS; // 16,777,216 elements
    const size_t bytes = N * sizeof(float);

    const int WARMUP_RUNS = 10;
    const int TIMING_RUNS = 100; 

    std::cout << "--- GELU Stable Timing Test ---" << std::endl;
    std::cout << "Array Size N: " << N << " (" << (double)bytes / (1024*1024*1024) << " GB)" << std::endl;

    std::vector<float> h_input(N);
    std::vector<float> h_output(N);

    for (int i = 0; i < N; ++i) {
        h_input[i] = 1.0f;
    }

    float *d_input, *d_output;
    checkCudaError(cudaMalloc(&d_input, bytes), "d_input allocation");
    checkCudaError(cudaMalloc(&d_output, bytes), "d_output allocation");

    checkCudaError(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice), "input copy H->D");

    const int THREADS_PER_BLOCK = 256;
    const int NUM_BLOCKS = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    std::cout << "Grid: " << NUM_BLOCKS << " blocks, " << THREADS_PER_BLOCK << " threads/block." << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout << "Warming up the GPU and Caches (" << WARMUP_RUNS << " runs)..." << std::endl;
    for (int i = 0; i < WARMUP_RUNS; ++i) {
        baseline_elementwise_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_output, d_input, N);
    }
    cudaDeviceSynchronize();
    checkCudaError(cudaGetLastError(), "warm-up kernel launch");

    float total_milliseconds = 0;
    std::cout << "Starting Stable Timing Loop (" << TIMING_RUNS << " runs)..." << std::endl;
    
    for (int i = 0; i < TIMING_RUNS; ++i) {
        cudaEventRecord(start);
        
        // Kernel launch
        baseline_elementwise_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_output, d_input, N);

        cudaEventRecord(stop);

        cudaEventSynchronize(stop); 

        float milliseconds_i = 0;
        cudaEventElapsedTime(&milliseconds_i, start, stop);
        total_milliseconds += milliseconds_i;
    }

    float average_milliseconds = total_milliseconds / TIMING_RUNS;
    
    std::cout << "\n--- Timing Results ---" << std::endl;
    std::cout << "Total execution time for " << TIMING_RUNS << " stable runs: " << total_milliseconds << " ms" << std::endl;
    std::cout << "**Average kernel execution time:** " << average_milliseconds * 1000.0f << " us" << std::endl;

    checkCudaError(cudaMemcpy(h_output.data(), d_output, bytes, cudaMemcpyDeviceToHost), "output copy D->H");

    const float expected_value = 0.84134475f; // GELU(1.0)=0.5⋅[1+erf(0.70710678)] => GELU(1.0)≈0.5⋅[1+0.68268949]
    if (std::abs(h_output[N/2] - expected_value) < 1e-5) {
        std::cout << "\nVerification Check: **PASSED** (midpoint value: " << h_output[N/2] << ")" << std::endl;
    } else {
        std::cout << "\nVerification Check: **FAILED** (midpoint value: " << h_output[N/2] << ", Expected: " << expected_value << ")" << std::endl;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}