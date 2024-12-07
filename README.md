# **DFT and FFT with CUDA Acceleration**

Discrete Fourier Transform (DFT) and Fast Fourier Transform (FFT) are fundamental techniques in signal processing, communications, and artificial intelligence for frequency domain analysis. Traditional serial implementations struggle with scalability and real-time demands for large datasets. By leveraging GPUs and CUDA, we optimized performance, minimized computation time and memory usage, and maximized GPU efficiency. This approach is particularly valuable for high-performance applications such as radar, medical imaging, and financial modeling.

---

## **Authors**
- **Tsung-Hsiang (Dawson) Ma**
- **Ya-Chi Liao**

---

## **Installation Instructions**
1. **Download and Extract**: Download the zip file and unzip it to your local device.  
2. **Set Up CMake**: Open CMake and select the unzipped folder as the source directory.  
3. **Create Build Directory**: Inside the source folder, create a `build` folder and select it as the build binary directory.  
4. **Configure**: Configure the project using **Visual Studio 16 2019** and set the architecture to `x64`.  
5. **Generate**: Generate the project files.  
6. **Open Solution File**: Navigate to the `build` folder and open the `.sln` file in **Visual Studio 2019**.  
7. **Run the Program**: Compile and execute the program within Visual Studio.  

---

## **Implementation Overview**
### **Discrete Fourier Transform (DFT)**
- **Benchmark (CPU)**: Baseline serial implementation.
- **CUDA Global Memory**: GPU acceleration using global memory.
- **CUDA Shared Memory**: Further optimized GPU implementation leveraging shared memory.

### **Fast Fourier Transform (FFT)**
- **Benchmark (CPU)**: Baseline serial implementation.
- **CUDA Global Memory**: GPU-accelerated FFT implementation.
- **CuFFT Library**: Reference implementation using NVIDIA's optimized FFT library.

---

## **Methodology**
### **DFT**
- **Thread Organization**  
  - Assign one thread per frequency component \( X[k] \).  
  - Use shared memory to minimize global memory access overhead.  

- **Kernel Design**  
  - Compute \( X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-2\pi knj/N} \) in each thread by iterating over \( x[n] \).  

- **Optimizations**  
  - **Coalesced Memory Access**: Improve data fetch efficiency from global memory.  
  - **Fast Math Operations**: Utilize CUDA's fast trigonometric functions.  

### **FFT**
- **Bit-Reversal Reordering**  
  - Each thread calculates the bit-reversed index for its assigned data point in parallel.  
  - GPU execution significantly reduces time complexity compared to serial reordering.  

- **Twiddle Factor Computation**  
  - Threads independently calculate twiddle factors for the butterfly operation.  
  - Parallelism eliminates inter-thread dependencies.  

- **Iterative Cooley-Tukey Algorithm**  
  - Parallel butterfly operations performed by threads ensure efficient computation.  
  - Pre-reordered input ensures coalesced memory access patterns for improved performance.  

---

## **Results**
| **N**                 | **16**     | **256**     | **1024**      |
|------------------------|------------|-------------|---------------|
| **DFT**               |            |             |               |
| Benchmark (CPU)       | 37.79 us   | 936 us      | 3.70 ms       |
| Global Memory         | 37.82 us   | 934 us      | 3.71 ms       |
| Shared Memory         | 38.50 us   | 917 us      | 3.52 ms       |
| **FFT**               |            |             |               |
| Benchmark (CPU)       | 41.7 us    | 7.74 ms     | 37.1 ms       |
| Global Memory         | 36.28 us   | 76.45 us    | 112.37 us     |
| CuFFT (Reference)     | 10.21 us   | 10.21 us    | 10.27 us      |

---

## **Prerequisites**
- **Hardware**: NVIDIA GPU supporting CUDA.  
- **Software**: CUDA Toolkit, Visual Studio 2019, and CMake.  
