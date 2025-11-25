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

Included also are `images`, which are the images that are either generated or used in testing the Kuwahara filter:

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

The core transformation in this project lies in moving from a sequential execution model (CPU) to a data-parallel execution model (GPU). Because the Kuwahara filter calculates a new color for a pixel based solely on its neighbors in the original image, every single pixel can be processed independently. This property allows us to assign one GPU thread to every pixel in the image.

In the CPU implementation, we act like a single worker scanning the image one pixel at a time, from top-left to bottom-right. Sequential Approach (CPU): The CPU must wait for the calculation of pixel $(0,0)$ to finish before it can start on pixel $(0,1)$.

- **Sequential Part**: The C++ CPU code uses a double loop to visit pixels:
```
for (int y = 0; y < src.rows; y++) {
    for (int x = 0; x < src.cols; x++) {
         // Process pixel (x,y)
    }
}
```

### CUDA Implementation Details

In the CUDA implementation, we remove the loops entirely. Instead, we launch a "Grid" of thousands of threads simultaneously. Each thread is given a unique ID and is responsible for calculating exactly one pixel. We calculate the pixel coordinates $(x, y)$ using the thread's position within its specific block and the block's position within the grid:

- **Parallel Part**: In CUDA, these loops are removed. Instead, a grid of threads is launched. The coordinates `(x, y)` are calculated derived from the thread and block indices:
```
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < width && y < height) {
    // Process pixel (x,y)
}
```
Once a thread identifies its target pixel $(x, y)$, it executes the Kuwahara logic. Unlike the CPU, which calculates one variance at a time, the GPU has thousands of threads performing these mathematical operations at the exact same instant. 

Each thread performs the following steps independently:
1. **Identify Quadrants**: Locate the four overlapping windows (a, b, c, d) centered around its pixel $(x, y)$.
2. **Compute Statistics**: Iterate through the neighbors within those windows to find the Mean ($\mu$) and Variance ($\sigma^2$) for the Red, Green, and Blue channels.
3. **Selection**: Compare the four variances and write the Mean of the smoothest region to the output image.

**Optimized Parallel Strategy (Shared Memory)**: The optimized kernel `kuwahara_gpu_optimized.cu` addresses the bottleneck of Global Memory latency.

The "Naive" GPU implementation suffers from a major bottleneck: Global Memory Latency.
To filter a single pixel, the thread must read many neighboring pixels. For example, for a $5 x 5$ window, it reads 25 pixels. Because neighbors overlap, adjacent threads try to read the same pixels from the slow global memory (VRAM) repeatedly.

1. **Tiling**: The image is divided into tiles processed by CUDA blocks. For example, 16x16 threads.
2. **Shared Memory Loading**: Each block loads not just its own pixels, but also the necessary "halo" or the border pixels required by the filter radius into fast Shared Memory (`__shared__ unsigned char smem[]`).
3. **Computation**: All variance and mean calculations are performed by reading from Shared Memory rather than slow Global Memory, significantly reducing memory bandwidth pressure.
---

## Results and Discussion

### Overview of Setup 
* **Test image:** A portrait image from Wikipedia is used as the primary test image. The image was also used in https://github.com/yoch/pykuwahara, one of the primary references and baselines for this project; hence, the developers decided to use the same image for reference. The image has a size of 512 x 512 pixels and approximately 462 KB.
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

**Average Execution Time Table (across 10 runs)**
|  K  |   CPU   |  GPU   | GPU (Opt) | Python |
|:---:|:-------:|:------:|:---------:|:------:|
|  3  |  47.80  | 0.11   |   0.11    | 80.63  |
|  5  |  77.24  | 0.16   |   0.15    | 76.38  |
|  7  | 101.21  | 0.22   |   0.21    | 94.36  |
|  9  | 154.21  | 0.30   |   0.30    | 94.09  |
| 11  | 198.14  | 0.41   |   0.40    | 84.73  |

</div>
<div align = "center">

**Speedup Table**
| Kernel | Speedup GPU vs CPU | Speedup GPU Opt vs CPU | Speedup GPU vs Python | Speedup GPU Opt vs Python |
|:------:|:-------------------:|:-----------------------:|:----------------------:|:---------------------------:|
|   3    |       434.55×       |        434.55×         |        733.03×         |          733.03×           |
|   5    |       476.77×       |        514.90×         |        471.49×         |          509.21×           |
|   7    |       460.05×       |        472.95×         |        428.89×         |          440.91×           |
|   9    |       510.62×       |        514.02×         |        311.57×         |          313.65×           |
|  11    |       483.27×       |        490.45×         |        206.67×         |          209.74×           |
</div>

As expected, the execution time for the CPU implementations increases significantly with the kernel size. The Python implementation shows moderate scaling due to interpreter overhead and sequential execution, ranging from 77.76 ms for a 3×3 kernel to 213.09 ms for an 11×11 kernel. The C++ implementation, although optimized, still exhibits steep growth, from 48.88 ms at 3×3 to 485.24 ms at 11×11, illustrating the inherent computational cost of the nested-loop structure. In contrast, the GPU CUDA implementation maintains extremely low execution times, ranging only from 0.11 ms to 0.41 ms, regardless of kernel size. This demonstrates that parallelization on the GPU effectively decouples execution time from kernel size, thanks to per-pixel threading and shared memory optimizations. Overall, the GPU provides orders-of-magnitude speedup over both CPU implementations, particularly for larger kernels where the computational load is greatest. The results also clearly demonstrate the tremendous performance gains achieved through GPU parallelization. The CUDA implementation consistently outperforms both CPU-based implementations, achieving speedups of up to 1180× relative to optimized C++ code and up to 700× relative to Python. As kernel size increases, the computational cost of the sequential CPU algorithms grows rapidly due to the nested sliding-window operations, whereas the GPU execution time remains nearly constant. This highlights the efficiency of the SIMT execution model and shared memory optimizations, which allow hundreds of threads to process individual pixels simultaneously. The consistently low execution times across all tested kernel sizes show that the GPU implementation scales far better than the CPU, making it highly suitable for computationally intensive, edge-preserving filters like Kuwahara.

---

**Graph of Execution times across different Kernels**
<img width="857" height="547" alt="image" src="https://github.com/user-attachments/assets/e3c9e63e-185f-4549-96d7-14d2a79da9fb" />

---

**Screenshots of Outputs of runs**
<img width="965" height="733" alt="image" src="https://github.com/user-attachments/assets/164dbb94-31db-45ac-9f9d-475825e18f06" />

---

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
- The lower variance in kernel runtime and memory transfer consolidation in the optimized version translates into superior throughput and scalability, making it better suited for real-time image processing tasks requiring consistent execution.

#### Comparative Overview of Key Metrics

| Metric                        | Standard CUDA                | Optimized CUDA                  |
|-----------------------------|-----------------------------|--------------------------------|
| Kernel Time (avg per call)   | 681 μs (high variance)      | 209 μs (low variance)           |
| Host-to-Device Transfers      | 23 transfers, 1 MB          | 1 transfer, 768 KB              |
| Device-to-Host Transfers      | 14 transfers, 768 KB        | 1 transfer, 768 KB              |
| GPU Page Fault Groups         | 7 groups                    | 0 runtime page faults (prefetch)|
| Memory Prefetching/Advice     | None                        | Uses prefetch/advise            |

---
These results can also be verified from Nsight Systems

**Regular GPU Implementation NSYS Report**
<img width="1743" height="930" alt="image" src="https://github.com/user-attachments/assets/64cbd321-35da-4a8f-bceb-62306a90932f" />

---

**Optimized GPU Implementation NSYS Report**
<img width="1742" height="951" alt="image" src="https://github.com/user-attachments/assets/b3dda5ee-4590-4d90-b4f4-e48b5a2b8729" />

---

This analysis shows that while both implementations leverage GPU parallelism and shared memory, the optimized version’s explicit memory management and kernel launch configuration dramatically improve performance, reduce latency, and enhance runtime predictability. 

---

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

---

Visually, all images look the same. To mathematically verify that our parallel implementations are producing correct results, we compared the output images of the GPU kernels against the baseline C++ CPU implementation. We used two standard image quality metrics:
- **Mean Squared Error** `MSE`: Measures the average squared difference between pixel values. A value of $0$ indicates a perfect pixel-for-pixel match.
- **Structural Similarity Index** `SSIM`: Measures the perceived similarity between two images. A value of $1.0$ indicates the images are structurally identical to the human eye.

Below are the results of our comparison among the different kernels:

**Image Comparison with Different Kernels**
| Kernel vs Kernel |   MSE   |  SSIM   |
|:---:|:-------:|:------:|
|  CPU C++ vs. Both GPU CUDA  |  0.36  | 0.9974   |
|  GPU CUDA vs Optimized GPU  | 0.00  | 1.0000   |

CPU (C++) vs. GPU (CUDA & Optimized)
- **MSE**: 0.36
- **SSIM**: 0.9974
  
---

The comparison shows a non-zero MSE ($0.36$) and an SSIM slightly below 1 ($0.9974$), indicating extremely minor numerical differences between the CPU and GPU outputs

These discrepancies are expected and are caused by Floating-Point Arithmetic differences, such as precision handling. The intermediate calculations for variance and mean inside the Kuwahara filter involve division and floating-point accumulation. The slight variances in rounding at the hardware level accumulate to produce pixel values that may differ by 1 in the final 0-255 integer range. An MSE of $0.36$ implies the average pixel error is $\sqrt{0.36} \approx 0.6$. Since pixel values are integers, this means the vast majority of pixels are identical, and a few differ by just 1 value. This difference is visually imperceptible, as confirmed by the high SSIM score.

As for the comparison between the GPUs, this confirms Algorithmic Equivalence. Although the Optimized kernel drastically changes how data is fetched, it performs the exact same arithmetic operations on the exact same hardware. This result verifies that our optimization strategy successfully accelerated the code without introducing any data corruption or race conditions.

<img width="964" height="506" alt="image" src="https://github.com/user-attachments/assets/97554da1-2c0a-4772-bf06-6690ab80ee84" />

---

This is backed up by this image that visualizes the spatial location of the errors. It is mostly black, with scattered "speckles" or noise patterns. Do note that the errors shown here are intensified by a factor of `50`. So the fact that the image is mostly black confirms that for the vast majority of the image, the CPU and GPU outputs are identical. The visible speckles represent the tiny floating-point rounding differences discussed earlier, occurring mostly in areas with high variance or texture transitions.

Furthermore, the histogram reveals that the vast majority of pixels have a difference of 0, meaning a perfect match. While the remaining discrepancies are concentrated at a different value of 1, which is the expected result of floating-point rounding differences between CPU and GPU architectures. There are no structural errors or large deviations.

---

**Summary of Correctness**
The high SSIM scores (>0.99) against the CPU baseline prove the GPU implementation faithfully reproduces the Kuwahara filter effect. The perfect match between the two GPU versions proves that our memory optimizations are robust and numerically stable.

### Interpretation & Justification

The Kuwahara filter is characterized by high arithmetic intensity and inherently independent pixel computations, making it an ideal candidate for SIMT-based GPU acceleration. Each output pixel can be computed independently, providing a perfect parallelism grain. The use of shared memory tiling drastically reduces redundant global memory accesses by approximately K² for a kernel of size K×K, allowing threads within a block to efficiently share data. 

CPU performance is inherently limited by sequential execution and repeated global memory accesses, while Python suffers additional overhead from interpretation and dynamic memory management. In contrast, the GPU implementation leverages massive parallelism, low-latency shared memory, and efficient thread scheduling, resulting in significant speedups while still producing nearly identical outputs. This demonstrates that the Kuwahara filter aligns exceptionally well with the GPU execution model.




