#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuComplex.h>

#define M_PI 3.14159265358979323846

__global__ void dftKernelShared(float* input_real, float* input_imag, float* output_real, float* output_imag, int N) {
    // Shared memory for input values
    extern __shared__ float shared_memory[];
    float* shared_real = shared_memory;
    float* shared_imag = shared_memory + N;

    int tid = threadIdx.x;
    int k = tid + blockIdx.x * blockDim.x;

    if (k < N) {
        // Load input into shared memory
        if (tid < N) {
            shared_real[tid] = input_real[tid];
            shared_imag[tid] = input_imag[tid];
        }
        __syncthreads(); // Ensure all threads load data before proceeding

        float sum_real = 0.0f;
        float sum_imag = 0.0f;

        for (int n = 0; n < N; ++n) {
            float angle = -2.0f * M_PI * k * n / N;
            float cos_val = cosf(angle);
            float sin_val = sinf(angle);

            sum_real += shared_real[n] * cos_val - shared_imag[n] * sin_val;
            sum_imag += shared_real[n] * sin_val + shared_imag[n] * cos_val;
        }

        // Store results in global memory
        output_real[k] = sum_real;
        output_imag[k] = sum_imag;
    }
}

// Host function to call the DFT kernel
void computeDFT_shared(float* h_input_real, float* h_input_imag, float* h_output_real, float* h_output_imag, int N) {
    // Allocate device memory
    float* d_input_real, * d_input_imag, * d_output_real, * d_output_imag;
    cudaMalloc(&d_input_real, N * sizeof(float));
    cudaMalloc(&d_input_imag, N * sizeof(float));
    cudaMalloc(&d_output_real, N * sizeof(float));
    cudaMalloc(&d_output_imag, N * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input_real, h_input_real, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_imag, h_input_imag, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel with shared memory
    int blockSize = 256; // Maximum number of threads per block
    int numBlocks = (N + blockSize - 1) / blockSize;
    size_t sharedMemorySize = 2 * N * sizeof(float); // Shared memory size for real and imaginary parts
    dftKernelShared << <numBlocks, blockSize, sharedMemorySize >> > (d_input_real, d_input_imag, d_output_real, d_output_imag, N);

    // Copy results back to host
    cudaMemcpy(h_output_real, d_output_real, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_imag, d_output_imag, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input_real);
    cudaFree(d_input_imag);
    cudaFree(d_output_real);
    cudaFree(d_output_imag);
}

int main() {
    // Example: DFT of a small array
    const int N = 1024;
    float h_input_real[N];
    float h_input_imag[N];
    for (int i = 0; i < N; i++) {
        h_input_real[i] = i + 1;
        h_input_imag[i] = 0;
    }

    float h_output_real[N];
    float h_output_imag[N];

    // Compute DFT
    computeDFT_shared(h_input_real, h_input_imag, h_output_real, h_output_imag, N);


    // Display results
    std::cout << "DFT Results:\n";
    for (int i = 0; i < N; ++i) {
        std::cout << "X[" << i << "] = " << h_output_real[i] << " + " << h_output_imag[i] << "i\n";
    }

    return 0;
}