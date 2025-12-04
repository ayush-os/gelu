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

    std::cout << "--- GELU Test (1000 Iterations) ---" << std::endl;
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
    const int NUM_ITERATIONS = 1000;

    std::cout << "Launching kernel " << NUM_ITERATIONS << " times with " << NUM_BLOCKS << " blocks and " << THREADS_PER_BLOCK << " threads/block." << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        baseline_elementwise_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_output, d_input, N);
    }

    cudaEventRecord(stop);

    cudaDeviceSynchronize();
    checkCudaError(cudaGetLastError(), "kernel launch");

    float total_milliseconds = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&total_milliseconds, start, stop);

    float average_milliseconds = total_milliseconds / NUM_ITERATIONS;
    
    std::cout << "Total execution time for " << NUM_ITERATIONS << " runs: " << total_milliseconds << " ms" << std::endl;
    std::cout << "Average kernel execution time: " << average_milliseconds * 1000.0f << " us" << std::endl;

    checkCudaError(cudaMemcpy(h_output.data(), d_output, bytes, cudaMemcpyDeviceToHost), "output copy D->H");

    const float expected_value = 0.84134475f; // GELU(1.0)=0.5⋅[1+erf(0.70710678)] => GELU(1.0)≈0.5⋅[1+0.68268949]
    if (std::abs(h_output[N/2] - expected_value) < 1e-5) {
        std::cout << "Verification Check: PASSED (midpoint value: " << h_output[N/2] << ")" << std::endl;
    } else {
        std::cout << "Verification Check: FAILED (midpoint value: " << h_output[N/2] << ", Expected: " << expected_value << ")" << std::endl;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}