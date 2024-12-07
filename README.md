# DFT and FFT with CUDA Acceleration


## Author
- Tsung-Hsiang (Dawson) Ma
- Ya-Chi Liao

## Install
1. Download the zip file, and unzip in your device.
2. Use cmake and select the unzip folder as source folder.
3. Create a folder named build inside the souce folder and select it as build binary directory.
4. Configure the file using Visual Studio 16 2019, and select x64 as computer archetecture.
5. Generate file.
6. Go into build folder and open sln file using visual studio 2019.
7. Run the program.

## Implementation
Discrete Fourier Transform (DFT)
- Benchmark (CPU)
- Acceleration with CUDA global memory
- Acceleration with CUDA shared memory
Fast Fourier Transform (FFT)
- Benchmark (CPU)
- Acceleration with CUDA global memory
- Implementation with CuFFT for reference

## Method


## Result
| N                    | 16       | 256      | 1024      |
|----------------------|----------|----------|-----------|
| DFT                  |          |          |           |
| Benchmark            | 37.79 us | 936 us   | 3.70 ms   |
| Global Mem           | 37.82 us | 934 us   | 3.71 ms   |
| Shared Mem           | 38.50 us | 917 us   | 3.52 ms   |
| FFT                  |          |          |           |
| Benchmark            | 41.7 us  | 7.74 ms  | 37.1 ms   |
| Global Mem           | 36.28 us | 76.45 us | 112.37 us |
| CuFFT(for reference) | 10.21 us | 10.21 us | 10.27 us  |

## Prerequisites

