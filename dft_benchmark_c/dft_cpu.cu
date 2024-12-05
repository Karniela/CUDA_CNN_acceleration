#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuComplex.h>

#define M_PI 3.14159265358979323846

void computeDFT_CPU(const float* input_real, const float* input_imag,
    float* output_real, float* output_imag, int N) {
    for (int k = 0; k < N; ++k) { // Iterate over each frequency bin
        double sum_real = 0.0;
        double sum_imag = 0.0;

        for (int n = 0; n < N; ++n) { // Sum over the input signal
            double angle = -2.0 * M_PI * k * n / N;
            double cos_val = cos(angle);
            double sin_val = sin(angle);

            sum_real += input_real[n] * cos_val - input_imag[n] * sin_val;
            sum_imag += input_real[n] * sin_val + input_imag[n] * cos_val;
        }

        output_real[k] = sum_real;
        output_imag[k] = sum_imag;
    }
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
    computeDFT_CPU(h_input_real, h_input_imag, h_output_real, h_output_imag, N);


    // Display results
    std::cout << "DFT Results:\n";
    for (int i = 0; i < N; ++i) {
        std::cout << "X[" << i << "] = " << h_output_real[i] << " + " << h_output_imag[i] << "i\n";
    }

    return 0;
}