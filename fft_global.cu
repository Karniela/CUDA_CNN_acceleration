
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuComplex.h>

#define THREADS_PER_BLOCK 256
#define PI 3.14159265358979323846

__device__ unsigned int bit_reverse(unsigned int n, unsigned int log2n) {
    unsigned int reversed = 0;
    for (unsigned int i = 0; i < log2n; i++) {
        reversed |= ((n >> i) & 1) << (log2n - 1 - i);
    }
    return reversed;
}

__global__ void bit_reversal_kernel(cuFloatComplex* data, unsigned int N, unsigned int log2n) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        unsigned int reversed_idx = bit_reverse(tid, log2n);
        if (tid < reversed_idx) {
            // Swap elements
            cuFloatComplex temp = data[tid];
            data[tid] = data[reversed_idx];
            data[reversed_idx] = temp;
        }
    }
}

__global__ void fft_kernel(cuFloatComplex* data, unsigned int N, unsigned int stage) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int m = 1 << stage;
    unsigned int m2 = m >> 1;  // Half of the stage size

    if (tid < N / 2) {
        unsigned int group = tid / m2;
        unsigned int pos = group * m + (tid % m2);

        cuFloatComplex w = make_cuFloatComplex(
            cosf(-2.0f * PI * (tid % m2) / m),
            sinf(-2.0f * PI * (tid % m2) / m));

        cuFloatComplex even = data[pos];
        cuFloatComplex odd = data[pos + m2];

        data[pos] = cuCaddf(even, cuCmulf(w, odd));
        data[pos + m2] = cuCsubf(even, cuCmulf(w, odd));
    }
}

void fft_cuda(cuFloatComplex* h_data, unsigned int N) {
    unsigned int log2n = (unsigned int)log2f((float)N);

    // Allocate memory on the GPU
    cuFloatComplex* d_data;
    cudaMalloc((void**)&d_data, N * sizeof(cuFloatComplex));
    cudaMemcpy(d_data, h_data, N * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

    // Step 1: Bit-reversal
    bit_reversal_kernel << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (d_data, N, log2n);

    // Step 2: Perform FFT stages
    for (unsigned int stage = 1; stage <= log2n; stage++) {
        fft_kernel << <(N / 2 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (d_data, N, stage);
    }

    // Copy the result back to the host
    cudaMemcpy(h_data, d_data, N * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_data);
}

int main() {
    // Example input size (power of 2)
    const unsigned int N = 1024;
    cuFloatComplex* h_data = (cuFloatComplex*)malloc(N * sizeof(cuFloatComplex));

    // Initialize input data with a sine wave or any test pattern
    for (unsigned int i = 0; i < N; i++) {
        h_data[i] = make_cuFloatComplex(sinf(2 * PI * i / N), 0.0f);  // Real values with imaginary = 0
    }

    printf("Input Data:\n");
    for (unsigned int i = 0; i < 8; i++) {
        printf("h_data[%d] = %.2f + %.2fi\n", i, cuCrealf(h_data[i]), cuCimagf(h_data[i]));
    }

    // Perform FFT
    fft_cuda(h_data, N);

    // Print the first few results
    printf("\nFFT Output:\n");
    for (unsigned int i = 0; i < 8; i++) {
        printf("h_data[%d] = %.2f + %.2fi\n", i, cuCrealf(h_data[i]), cuCimagf(h_data[i]));
    }

    // Free host memory
    free(h_data);

    return 0;
}
