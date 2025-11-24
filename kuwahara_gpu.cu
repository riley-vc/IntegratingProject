
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>

using namespace cv;
using namespace std;

// macro for error
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPU Error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// cuda kernel
__global__ void kuwaharaShared(unsigned char* src, unsigned char* dst, int width, int height, int radius) {
    extern __shared__ unsigned char smem[];

    int sm_width = blockDim.x + 2 * radius;
    int tile_top_left_x = blockIdx.x * blockDim.x - radius;
    int tile_top_left_y = blockIdx.y * blockDim.y - radius;

    int total_sm_pixels = sm_width * (blockDim.y + 2 * radius);
    int threads_per_block = blockDim.x * blockDim.y;
    int thread_id_linear = threadIdx.y * blockDim.x + threadIdx.x;

    // load global memory to shared memory
    for (int i = thread_id_linear; i < total_sm_pixels; i += threads_per_block) {
        int local_y = i / sm_width;
        int local_x = i % sm_width;
        int global_y = tile_top_left_y + local_y;
        int global_x = tile_top_left_x + local_x;
        int sm_idx = i * 3;

        if (global_x >= 0 && global_x < width && global_y >= 0 && global_y < height) {
            int global_idx = (global_y * width + global_x) * 3;
            smem[sm_idx] = src[global_idx];
            smem[sm_idx+1] = src[global_idx+1];
            smem[sm_idx+2] = src[global_idx+2];
        } else {
            // Padding with zeros (or could replicate border)
            smem[sm_idx] = 0; smem[sm_idx+1] = 0; smem[sm_idx+2] = 0;
        }
    }
    
    __syncthreads();

    // Compute Kuwahara
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int sm_center_x = threadIdx.x + radius;
        int sm_center_y = threadIdx.y + radius;
        
        float minVariance = 1e9;
        float bestMean[3] = {0, 0, 0};
        
        // 4 sub-windows relative to the center pixel
        int ranges[4][4] = {
            {-radius, 0, -radius, 0}, // Top left
            {0, radius, -radius, 0},  // T op right
            {-radius, 0, 0, radius},  // Bottom left
            {0, radius, 0, radius}    // Bottom right
        };

        for (int k = 0; k < 4; k++) {
            float s[3] = {0,0,0};
            float ss[3] = {0,0,0};
            int count = 0;

            for (int dy = ranges[k][2]; dy <= ranges[k][3]; dy++) {
                for (int dx = ranges[k][0]; dx <= ranges[k][1]; dx++) {
                    int sy = sm_center_y + dy;
                    int sx = sm_center_x + dx;
                    int idx = (sy * sm_width + sx) * 3;
                    
                    for(int c=0; c<3; c++) {
                        float val = smem[idx+c];
                        s[c] += val;
                        ss[c] += val * val;
                    }
                    count++;
                }
            }
            
            float mean[3], var[3], totVar = 0;
            for(int c=0; c<3; c++) {
                mean[c] = s[c] / count;
                var[c] = (ss[c] / count) - (mean[c] * mean[c]);
                totVar += var[c];
            }

            if (totVar < minVariance) {
                minVariance = totVar;
                for(int c=0; c<3; c++) bestMean[c] = mean[c];
            }
        }
        
        int outIdx = (y * width + x) * 3;
        dst[outIdx] = (unsigned char)bestMean[0];
        dst[outIdx+1] = (unsigned char)bestMean[1];
        dst[outIdx+2] = (unsigned char)bestMean[2];
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: ./kuwahara_gpu <image> <size>\n");
        return -1;
    }
    
    string inputPath = argv[1];
    int kernelSize = atoi(argv[2]);
    int radius = kernelSize / 2; // Derived radius

    Mat img = imread(inputPath, IMREAD_COLOR);
    if (img.empty()) {
        fprintf(stderr, "Error reading image\n");
        return -1;
    }
    
    int width = img.cols;
    int height = img.rows;
    size_t size = width * height * 3;

    unsigned char *d_src, *d_dst;
    cudaMallocManaged(&d_src, size);
    cudaMallocManaged(&d_dst, size);
    
    // copy data to unified memory
    if (img.isContinuous()) {
        memcpy(d_src, img.data, size);
    } else {
        for (int i = 0; i < height; ++i) 
            memcpy(d_src + i * width * 3, img.ptr(i), width * 3);
    }
    
    cudaDeviceSynchronize();

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    // Shared memory needs to hold the block + padding on all sides
    size_t smemSize = (block.x + 2*radius) * (block.y + 2*radius) * 3;

    kuwaharaShared<<<grid, block, smemSize>>>(d_src, d_dst, width, height, radius);
    cudaDeviceSynchronize();

    // timing process
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    kuwaharaShared<<<grid, block, smemSize>>>(d_src, d_dst, width, height, radius);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // output for python:
    cout << "GPU_TIME:" << milliseconds << endl;

    // copy back and save
    Mat result(height, width, CV_8UC3);
    memcpy(result.data, d_dst, size);
    
    // write output image
    string outputPath = "gpu-cuda-output.jpg";
    imwrite(outputPath, result);
    cout << "Saved filtered image to: " << outputPath << endl;
    
    cudaFree(d_src);
    cudaFree(d_dst);
    return 0;
}
