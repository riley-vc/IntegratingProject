# IntegratingProject

# Parallelizing the Kuwahara Filter: A SIMT Implementation

**Course:** CSC612M - Advanced Computer Architecture  
**Group:** 5  

## Group Members
* **Carandang, Matthew Ryan**
* **Veracruz, Sean Riley**
* **Yap, Rafael Subo**

---

## Abstract
The **Kuwahara filter** is an advanced, non-linear, edge-preserving smoothing filter used in image processing to reduce noise while maintaining sharp edges. Unlike standard Gaussian blurs, the Kuwahara filter calculates the mean and variance of four overlapping sub-regions for every single pixel, making it computationally exhaustive on sequential CPUs.

This project focuses on the implementation and performance analysis of the Kuwahara Filter, a non-linear smoothing filter used in image processing for noise reduction that preserves edges. The primary objective is to demonstrate the computational advantages of parallel computing architectures over traditional sequential methods.

The project implements and compares four different kernels to apply the filter to an image:

- **CPU Kernel (Python)**: A baseline high-level implementation.

- **CPU Kernel (C++)**: A sequential, compiled implementation for a stronger baseline.

- **GPU Kernel (CUDA C++)**: A parallel implementation utilizing Global/Unified Memory.

- **Optimized GPU Kernel (CUDA C++)**: An optimized parallel implementation utilizing Shared Memory and memory prefetching.

---

## Repository Details.

The repository includes a multitude of files that are important for the project. The table below shows each file and what it contains:

| File | Description|
|--------|---------------|
| `IntegratingProject.ipynb` | Contains all kernel implementations of the Kuwahara filter. Simulates the environment to run and compare the different kernels through Google Colab |
| `kuwahara.cpp` | Contains the `C++` implementation of the Kuwahara filter |
| `kuwahara.py` | Contains the `Python` implementation of the Kuwahara filter |
| `kuwahara_gpu.cu` | Contains the `CUDA C++` implementation of the Kuwahara filter |
| `kuwahara_gpu_optimized.cu` | Contains the **Optimized** `CUDA C++` implementation of the kuwahara filter |
| `kuwahara_gpu.nsys-rep` | Contains the report for the `CUDA C++` implementation of the kuwahara filter that can be viewed using Nvidia NSight Systems |
| `kuwahara_gpu_optimized.nsys-rep` | Contains the report for the **Optimized** `CUDA C++` implementation of the kuwahara filter that can be viewed using Nvidia NSight Systems |

Included also is an `images` folder where it contains the following images:

| File | Description|
|--------|---------------|
| `test.jpg` | The base image without filter. This is the image on which all kernels will be applying the filter on. |
| `cpu-cpp-output.jpg` | The output image of the `C++` Kernel with the filter applied. |
| `cpu-py-output.jpg` | The output image of the `Python` Kernel with the filter applied. |
| `gpu-cuda-output.jpg` | The output image of the `CUDA C++` Kernel with the filter applied. |
| `gpu-optimizedcuda-output.jpg` | The output image of the **Optimized** `Cuda C++` Kernel with the filter applied. |

---

## Methodology

### Algorithm: Kuwahara Filter

The Kuwahara filter works by calculating the mean and variance of color values in four overlapping sub-windows (quadrants) surrounding a target pixel. For every pixel $(x, y)$ in the image:

1. Divide the neighborhood into four rectangular regions overlapping at $(x, y)$.
2. Calculate the arithmetic mean and standard deviation (variance) of the pixel intensities for each region.
3. Select the region with the lowest variance (indicating the most homogeneous texture).
4. Assign the mean color of that selected region to the central pixel $(x, y)$.

### Implementations

- **Sequential (CPU)**: The C++ implementation (`kuwahara_cpu.cpp`) uses the OpenCV library. It iterates sequentially through every pixel using nested `for` loops (rows `y`, columns `x`). For each pixel, it iterates through the four sub-quadrants to calculate variance and selects the best mean. `copyMakeBorder` is used to handle edge padding.
- **Parallel (GPU - CUDA)**: The CUDA implementations (`kuwahara_gpu.cu` and `kuwahara_gpu_optimized.cu`) offload the per-pixel calculation to the GPU. The image data is managed using Unified Memory (`cudaMallocManaged`), allowing access from both CPU and GPU.

### CUDA Implementation Details

The core transformation from sequential to parallel lies in the independence of the pixel calculations. Since the output value of a pixel $(x, y)$ depends only on its neighbors in the input image (read-only), every pixel can be processed simultaneously.

Highlight of the Conversion:

- **Sequential Part**: The C++ CPU code uses a double loop to visit pixels:
```
for (int y = 0; y < src.rows; y++) {
    for (int x = 0; x < src.cols; x++) {
         // Process pixel (x,y)
    }
}
```
- **Parallel Part**: In CUDA, these loops are removed. Instead, a grid of threads is launched. The coordinates `(x, y)` are calculated derived from the thread and block indices:
```
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < width && y < height) {
    // Process pixel (x,y)
}
```

**Optimized Parallel Strategy (Shared Memory)**: The optimized kernel `kuwahara_gpu_optimized.cu` addresses the bottleneck of Global Memory latency.

1. **Tiling**: The image is divided into tiles processed by CUDA blocks (e.g., 16x16 threads).
2. **Shared Memory Loading**: Each block loads not just its own pixels, but also the necessary "halo" (border pixels required by the filter radius) into fast Shared Memory (`__shared__ unsigned char smem[]`).
3. **Computation**: All variance and mean calculations are performed by reading from Shared Memory rather than slow Global Memory, significantly reducing memory bandwidth pressure.
---

## Results and Discussion
### Overview of Setup 
* **Test image:** A portrait image from Wikipedia is used as the primary test image. The image was also used in https://github.com/yoch/pykuwahara, one of the primary references and baselines for this project, hence the developers decided to use the same image for reference. The image has a size of 512 x 512 pixels and approximately 462 KB.
  <div align ="center">
  <img width="512" height="512" alt="Lenna_(test_image)" src="https://github.com/user-attachments/assets/f9c3d889-7ac3-44ed-81f9-c0e8c80ab6dc" />
  </div>
* **Kernel sizes evaluated**: [3, 5, 7, 9, 11]
* **Hardware specifications**: The testing and data collection were performed on the Google Colab environment, providing a standardized cloud environment and set of hardware for testing
  * CPU: Intel(R) Xeon(R) CPU @ 2.00GHz (2 Cores)
  * GPU: NVIDIA Tesla T4 — 15 GB VRAM, Driver 550.54.15, CUDA 12.4 (nvcc 12.5)
* **Execution time tools**:
  * C++ implementation: *std::chrono::high_resolution_clock*
  * Python implementation: *time.perf_counter()*
  * CUDA GPU implementation: *cudaEventRecord(start), kernel launch, cudaEventRecord(stop), cudaEventElapsedTime(&ms, start, stop)*
### Execution Time Summary and Performance vs. Kernel Size

<div align = "center">

**Execution Time Table**
| Kernel | CPU C++ (ms) | GPU CUDA (ms) | Python (ms) |
|:------:|:------------:|:-------------:|:-----------:|
| 3      | 48.88        | 0.11          | 77.76       |
| 5      | 83.72        | 0.16          | 72.80       |
| 7      | 142.15       | 0.22          | 103.31      |
| 9      | 274.89       | 0.31          | 153.33      |
| 11     | 485.24       | 0.41          | 213.09      |

</div>
<div align = "center">

**Speedup Table**
| Kernel | CPU C++ → CUDA | Python → CUDA |
|-------:|---------------:|--------------:|
| 3      | 444×          | 707×          |
| 5      | 523×          | 455×          |
| 7      | 646×          | 470×          |
| 9      | 887×          | 495×          |
| 11     | 1183×         | 520×          |
</div>

As expected, the execution time for the CPU implementations increases significantly with the kernel size. The Python implementation shows moderate scaling due to interpreter overhead and sequential execution, ranging from 77.76 ms for a 3×3 kernel to 213.09 ms for an 11×11 kernel. The C++ implementation, although optimized, still exhibits steep growth, from 48.88 ms at 3×3 to 485.24 ms at 11×11, illustrating the inherent computational cost of the nested-loop structure. In contrast, the GPU CUDA implementation maintains extremely low execution times, ranging only from 0.11 ms to 0.41 ms, regardless of kernel size. This demonstrates that parallelization on the GPU effectively decouples execution time from kernel size, thanks to per-pixel threading and shared memory optimizations. Overall, the GPU provides orders-of-magnitude speedup over both CPU implementations, particularly for larger kernels where the computational load is greatest. The results also clearly demonstrate the tremendous performance gains achieved through GPU parallelization. The CUDA implementation consistently outperforms both CPU-based implementations, achieving speedups of up to 1180× relative to optimized C++ code and up to 700× relative to Python. As kernel size increases, the computational cost of the sequential CPU algorithms grows rapidly due to the nested sliding-window operations, whereas the GPU execution time remains nearly constant. This highlights the efficiency of the SIMT execution model and shared memory optimizations, which allow hundreds of threads to process individual pixels simultaneously. The consistent low execution times across all tested kernel sizes show that the GPU implementation scales far better than the CPU, making it highly suitable for computationally intensive, edge-preserving filters like Kuwahara.

### System-Level Analysis

Profiling both the standard and optimized CUDA implementations of the Kuwahara filter reveals key insights into kernel efficiency, memory management, and system overheads, highlighting the impact of optimization strategies on GPU performance.

#### Kernel Execution Performance

- The kernel execution for the standard implementation (`kuwaharaShared`) averages 681 μs per call with notable variance (stddev ~583 μs), reflecting less consistent performance. In contrast, the optimized kernel reduces average execution time to approximately 209 μs with very low variance (stddev ~972 ns), indicating both faster and more predictable processing.
- This reduction is primarily driven by improved thread and block configurations, efficient use of shared memory, and minimized memory access latency in the optimized version.

#### Memory Transfer and Unified Memory Paging

- The standard implementation performs frequent unified memory operations, including 23 host-to-device transfers and 14 device-to-host transfers, moving over 1 MB and 768 KB respectively. The kernel also incurs 7 GPU page fault groups increased latency.
- Conversely, the optimized version consolidates memory transfers into a single large transfer in each direction (~768 KB), greatly reducing the overhead associated with page migrations and memory transfers. Prefetching via `cudaMemPrefetchAsync` and memory advice (`cudaMemAdvise`) proactively reduces runtime page faults and enhances data locality on the device, contributing to lower kernel runtime and increased efficiency.

#### Unified Memory Management and API Overheads

- Both implementations experience significant overhead from unified memory allocation (`cudaMallocManaged`), which dominates the CUDA API call time. However, the optimized code’s explicit prefetch and advise calls reduce the latency impact during kernel execution, enabling better overlap and fewer synchronization stalls compared to the standard code.
- Kernel launch overheads and synchronization times are lower and more stable in the optimized implementation, reflecting smoother GPU workflow.

#### System-Level Behavior and Resource Utilization

- System call profiling shows high time spent in `poll` and `ioctl`, characteristic of GPU-bound programs waiting on kernel completions. This behavior is intrinsic to workloads with GPU offload and does not indicate inefficiency.
- The lower variance in kernel runtime and memory transfer consolidation in the optimized version translate into superior throughput and scalability, making it better suited for real-time image processing tasks requiring consistent execution.

#### Comparative Overview of Key Metrics

| Metric                        | Standard CUDA                | Optimized CUDA                  |
|-----------------------------|-----------------------------|--------------------------------|
| Kernel Time (avg per call)   | 681 μs (high variance)      | 209 μs (low variance)           |
| Host-to-Device Transfers      | 23 transfers, 1 MB          | 1 transfer, 768 KB              |
| Device-to-Host Transfers      | 14 transfers, 768 KB        | 1 transfer, 768 KB              |
| GPU Page Fault Groups         | 7 groups                    | 0 runtime page faults (prefetch)|
| Memory Prefetching/Advice     | None                        | Uses prefetch/advise            |


---

This analysis shows that while both implementations leverage GPU parallelism and shared memory, the optimized version’s explicit memory management and kernel launch configuration dramatically improve performance, reduce latency, and enhance runtime predictability. 


### Image Quality Comparison
<div align = "center">
 
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/4a9165f5-9961-4e4a-a1de-cb7603e261a9" width="150"/></td>
    <td><img src="https://github.com/user-attachments/assets/83cc130d-446a-4876-a7c9-80a5fa3cc358" width="150"/></td>
    <td><img src="https://github.com/user-attachments/assets/fa80e7c7-a3ee-4722-a544-2cbfd2eeba74" width="150"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/0c3d12ee-f901-405a-a0fe-2f9ad3cb8a95" width="150"/></td>
    <td><img src="https://github.com/user-attachments/assets/d42da223-c6cd-462b-b146-5d46d68fa269" width="150"/></td>
  </tr>
</table>
</div>
At a kernel size of 7×7, all three implementations—CUDA, C++, and Python—produce virtually identical outputs. The filter effectively smooths noise while preserving edges, and any minor differences are imperceptible, arising only from rounding or implementation-specific pixel calculations. This confirms the correctness of the GPU-accelerated version relative to the CPU baselines.

### Interpretation & Justification

The Kuwahara filter is characterized by high arithmetic intensity and inherently independent pixel computations, making it an ideal candidate for SIMT-based GPU acceleration. Each output pixel can be computed independently, providing a perfect parallelism grain. The use of shared memory tiling drastically reduces redundant global memory accesses by approximately K² for a kernel of size K×K, allowing threads within a block to efficiently share data. 

CPU performance is inherently limited by sequential execution and repeated global memory accesses, while Python suffers additional overhead from interpretation and dynamic memory management. In contrast, the GPU implementation leverages massive parallelism, low-latency shared memory, and efficient thread scheduling, resulting in significant speedups while still producing nearly identical outputs. This demonstrates that the Kuwahara filter aligns exceptionally well with the GPU execution model.




