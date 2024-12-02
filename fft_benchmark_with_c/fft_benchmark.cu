#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265358979323846

// Define a struct for complex numbers
typedef struct {
    double real;
    double imag;
} Complex;

// Function to print complex numbers
void printComplexArray(Complex* arr, int n) {
    for (int i = 0; i < n; i++) {
        printf("(%f, %f)\n", arr[i].real, arr[i].imag);
    }
}

// Function to perform bit reversal reordering
void bitReversal(Complex* a, int n) {
    int bits = log2(n); // Number of bits needed to represent indices
    for (int i = 0; i < n; i++) {
        int reversed = 0;
        int x = i;
        for (int j = 0; j < bits; j++) {
            reversed = (reversed << 1) | (x & 1);
            x >>= 1;
        }
        if (i < reversed) {
            // Swap elements at index i and reversed
            Complex temp = a[i];
            a[i] = a[reversed];
            a[reversed] = temp;
        }
    }
}

// Function to perform iterative FFT with bit reversal
void fft(Complex* a, int n) {
    if ((n & (n - 1)) != 0) {
        fprintf(stderr, "Error: FFT input size must be a power of 2.\n");
        exit(EXIT_FAILURE);
    }

    // Perform bit reversal reordering
    bitReversal(a, n);

    // Iterative Cooley-Tukey FFT
    for (int len = 2; len <= n; len *= 2) {
        double angle = -2.0 * PI / len;
        Complex wLen = { cos(angle), sin(angle) };

        for (int i = 0; i < n; i += len) {
            Complex w = { 1.0, 0.0 };
            for (int j = 0; j < len / 2; j++) {
                Complex u = a[i + j];
                Complex t = { w.real * a[i + j + len / 2].real - w.imag * a[i + j + len / 2].imag,
                             w.real * a[i + j + len / 2].imag + w.imag * a[i + j + len / 2].real };

                a[i + j].real = u.real + t.real;
                a[i + j].imag = u.imag + t.imag;
                a[i + j + len / 2].real = u.real - t.real;
                a[i + j + len / 2].imag = u.imag - t.imag;

                // Update w = w * wLen
                double tempReal = w.real * wLen.real - w.imag * wLen.imag;
                w.imag = w.real * wLen.imag + w.imag * wLen.real;
                w.real = tempReal;
            }
        }
    }
}

int main() {
    // Example input: signal array
    int n = 8; // Length of the signal (must be a power of 2)
    Complex signal[] = {
        {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}
    };

    printf("Input signal:\n");
    printComplexArray(signal, n);

    // Perform FFT
    fft(signal, n);

    printf("\nFFT output:\n");
    printComplexArray(signal, n);

    return 0;
}
