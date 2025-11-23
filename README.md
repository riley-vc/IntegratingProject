# IntegratingProject

# Parallelizing the Kuwahara Filter: A SIMT Implementation

**Course:** CSC612M - Advanced Computer Architecture  
**Group:** 5  

## 👥 Team Members
* **Carandang, Matthew Ryan**
* **Veracruz, Sean Riley**
* **Yap, Rafael Subo**

* ---
## Abstract
The **Kuwahara filter** is an advanced, non-linear, edge-preserving smoothing filter used in image processing to reduce noise while maintaining sharp edges. Unlike standard Gaussian blurs, the Kuwahara filter calculates the mean and variance of four overlapping sub-regions for every single pixel, making it computationally exhaustive on sequential CPUs.

This project implements the Kuwahara filter from scratch using the **GPU/CUDA (SIMT)** paradigm. By assigning one CUDA thread to each output pixel and utilizing **Shared Memory tiling**, we aim to overcome the memory bandwidth bottlenecks inherent in the algorithm and demonstrate significant speedups compared to sequential CPU implementations.

## Project Goals
1.  **Parallelize the Sliding Window:** Convert the $O(N^2)$ nested-loop structure into a data-parallel SIMT kernel.
2.  **Optimize Memory Access:** Implement **Shared Memory Tiling** with halo (border) pixel management to minimize global memory latency.
3.  **Benchmark:** Quantify the performance gap (execution time in ms) between the sequential CPU baseline and the optimized GPU kernel.

---

## Repository Details.

The repository includes a comprehensive Jupyter Notebook (IntegratingProject.ipynb) that orchestrates the entire project:

* **Environment Setup**: Automatically installs dependencies (libopencv-dev) and configures the NVCC compiler.

* **Compilation**: Compiles both the C++ baseline and the CUDA kernel directly within the notebook.

* **Execution & Timing**: Runs all three implementations on the same input image and captures execution time.

* **Visualization**: Displays the input, noisy image, and the filtered results side-by-side for visual quality comparison.

* **Benchmarking**: Generates comparative graphs showing the execution time vs. kernel radius for Python, C++, and CUDA.
---

## C++ and Python Implementation Details

To accurately measure the performance gains of our GPU kernel, we implemented the Kuwahara filter in two standard CPU-based environments. These serve as our control group for benchmarking.

1. Python Implementation 

* Serves as the algorithmic proof-of-concept and correctness verification.

* Libraries: NumPy for matrix operations and OpenCV (cv2) for image I/O.

* Methodology:

  * Implements the filter using standard nested loops to iterate over every pixel.

  * Calculates the mean and variance for the four sub-quadrants (top-left, top-right, bottom-left, bottom-right) for each pixel.

  * Python is excellent for prototyping. But this implementation highlights the significant overhead of interpreted loops for per-pixel image processing tasks, making it the slowest of the three.

2. C++ Implementation

* Provides a "fair" high-performance CPU baseline. Unlike Python, C++ is compiled and optimized, which represents how a standard CPU-based image processing application would typically run.

* Methodology:

* Written in standard C++11.

  * Utilizes the exact same sliding window logic as the CUDA kernel but runs sequentially on the host CPU.

  * Compiled with optimization flags (e.g., -O3) to ensure maximum CPU efficiency.

  * The C++ implementation is still bound by the sequential nature of the CPU processing one pixel (or small groups of pixels) at a time, scaling poorly as image resolution or kernel size increases.


## CUDA Implementation Details

### The Algorithm
For every pixel $(x, y)$ in the image, the filter considers a window of size $W$ (e.g., $5 \times 5$, $7 \times 7$). The window is divided into four overlapping sub-regions (quadrants). The algorithm:
1.  Calculates the **mean** and **variance** of intensity for each of the four quadrants.
2.  Identifies the quadrant with the **minimum variance** (the most homogeneous region).
3.  Sets the central pixel's value to the **mean** of that minimum-variance quadrant.

### Parallel Strategy (SIMT)
* **Grid Topology:** A 2D grid of thread blocks is launched.
* **Granularity:** One CUDA thread is responsible for computing one output pixel.
* **Shared Memory Optimization:**
    * Instead of every thread reading 25+ pixels from global memory (for a 5x5 window), threads in a block cooperate to load a "tile" of the image into on-chip Shared Memory.
    * This tile includes a **"halo"** (apron) of extra pixels required for the boundary conditions of the thread block.
    * All statistical calculations are performed by reading from the blazing-fast shared memory.

---

## Prerequisites

To build and run this project, you need:
* **Hardware:** NVIDIA GPU (Compute Capability 5.0 or higher recommended).
* **Compiler:** `nvcc` (NVIDIA CUDA Compiler) and `g++`.
* **Libraries:** * OpenCV (C++ core) - *Used for Image I/O only*.
    * (Optional) `stb_image` if OpenCV is not used.

## Results and Discussion



