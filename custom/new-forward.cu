#include <cmath>
#include <iostream>
#include "cuda_fp16.h"
#include "gpu-new-forward.h"

#define TILE_WIDTH 20

#define MAX_CM_SIZE 4096
__constant__ float d_mask[MAX_CM_SIZE];

// #define NUM_STREAMS 10
// cudaStream_t stream[NUM_STREAMS];
// int input_size_stream, output_size_stream;

__global__ void conv_forward_kernel_cm_and_tuning(float * __restrict__ output, const float * __restrict__ input, const float * __restrict__ mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a
    int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
    int tx = threadIdx.x, ty = threadIdx.y;

    int W_grid = (Width_out - 1)/TILE_WIDTH + 1;

    int blockHeight = (bz / W_grid) * TILE_WIDTH + ty;
    int blockWidth = (bz % W_grid) * TILE_WIDTH + tx;

    float acc = 0.0;
    int c, p, q;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) d_mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    #pragma unroll
    for (c = 0; c < Channel; c++) {
        #pragma unroll
        for (p = 0; p < 7; p++) {
            #pragma unroll
            for (q = 0; q < 7; q++) {
                acc += in_4d(bx, c, blockHeight + p, blockWidth + q) * mask_4d(by, c, p, q);
            }
        }
    }
    if (blockHeight < Height_out && blockWidth < Width_out)
        out_4d(bx, by, blockHeight, blockWidth) = acc;

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

__global__ void conv_forward_kernel_cm(float * output, const float * input, const float * mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a
    int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
    int tx = threadIdx.x, ty = threadIdx.y;

    int W_grid = (Width_out - 1)/TILE_WIDTH + 1;

    int blockHeight = (bz / W_grid) * TILE_WIDTH + ty;
    int blockWidth = (bz % W_grid) * TILE_WIDTH + tx;

    //half acc = 0.0;
    float acc = 0.0;
    int c, p, q;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) d_mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    for (c = 0; c < Channel; c++) {
        for (p = 0; p < K; p++) {
            for (q = 0; q < K; q++) {
                acc += in_4d(bx, c, blockHeight + p, blockWidth + q) * mask_4d(by, c, p, q);
            }
        }
    }
    if (blockHeight < Height_out && blockWidth < Width_out)
        out_4d(bx, by, blockHeight, blockWidth) = acc;

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}
__global__ void conv_forward_kernel_fp16(float * output, const float * input, const float * mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a
    int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
    int tx = threadIdx.x, ty = threadIdx.y;

    int W_grid = (Width_out - 1)/TILE_WIDTH + 1;

    int blockHeight = (bz / W_grid) * TILE_WIDTH + ty;
    int blockWidth = (bz % W_grid) * TILE_WIDTH + tx;

    half acc = 0.0;
    int c, p, q;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    for (c = 0; c < Channel; c++) {
        for (p = 0; p < K; p++) {
            for (q = 0; q < K; q++) {
                acc += __float2half(in_4d(bx, c, blockHeight + p, blockWidth + q)) * __float2half(mask_4d(by, c, p, q));
            }
        }
    }
    if (blockHeight < Height_out && blockWidth < Width_out)
        out_4d(bx, by, blockHeight, blockWidth) = __half2float(acc);

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

__global__ void conv_forward_kernel_base(float * output, const float * input, const float * mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a
    int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
    int tx = threadIdx.x, ty = threadIdx.y;

    int W_grid = (Width_out - 1)/TILE_WIDTH + 1;

    int blockHeight = (bz / W_grid) * TILE_WIDTH + ty;
    int blockWidth = (bz % W_grid) * TILE_WIDTH + tx;

    //half acc = 0.0;
    float acc = 0.0;
    int c, p, q;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    for (c = 0; c < Channel; c++) {
        for (p = 0; p < K; p++) {
            for (q = 0; q < K; q++) {
                acc += in_4d(bx, c, blockHeight + p, blockWidth + q) * mask_4d(by, c, p, q);
            }
        }
    }
    if (blockHeight < Height_out && blockWidth < Width_out)
        out_4d(bx, by, blockHeight, blockWidth) = acc;

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}
	
__host__ void GPUInterface::conv_forward_gpu_prolog(float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
    
    // malloc/memcpy sizes
    int output_size = Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float);
    int input_size = Batch * Channel * Height * Width * sizeof(float);
    int mask_size = Map_out * Channel * K * K * sizeof(float);
    
    // set stream async memcpy sizes
    // output_size_stream = output_size / NUM_STREAMS;
    // input_size_stream = input_size / NUM_STREAMS;
        
    // #pragma unroll
    // for (int i = 0; i < NUM_STREAMS; i++) {
    //     cudaStreamCreate(&stream[i]);
    // }

    // cudaHostRegister((void *)host_input, input_size, cudaHostRegisterDefault);
    // cudaHostRegister((void *)host_output, output_size, cudaHostRegisterDefault);

    cudaMalloc((void **) device_output_ptr, output_size);
    cudaMalloc((void **) device_input_ptr, input_size);
    // cudaMalloc((void **) device_mask_ptr, mask_size);
    cudaMemcpyToSymbol(d_mask, host_mask, mask_size, 0, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice);
    // cudaMemcpy(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice);

    // #pragma unroll
    // for (int i = 0; i < NUM_STREAMS; i++) {
    //     cudaMemcpyAsync((*device_input_ptr) + (i * (input_size_stream / sizeof(float))), host_input + (i * (input_size_stream / sizeof(float))), input_size_stream, cudaMemcpyHostToDevice, stream[i]);
    //     conv_forward_kernel_cm_and_tuning<<<dimGrid, dimBlock, 0, stream[i]>>>((*device_output_ptr) + (i * (output_size_stream / sizeof(float))), (*device_input_ptr) + (i * (input_size_stream / sizeof(float))), *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
    //     cudaMemcpyAsync(host_output + (i * (output_size_stream / sizeof(float))), (*device_output_ptr) + (i * (output_size_stream / sizeof(float))), output_size_stream, cudaMemcpyDeviceToHost, stream[i]);
    // }

    // cudaHostUnregister((void *)host_input);
    // cudaHostUnregister((void *)host_output);
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    int W_grid = ceil(1.0 * Width/TILE_WIDTH);
    int H_grid = ceil(1.0 * Height/TILE_WIDTH);
    int Z = W_grid * H_grid;

    int Width_out = Width - K + 1;
    int Height_out = Height - K + 1;
    
    dim3 dimGrid(Batch, Map_out, Z);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    // #pragma unroll
    // for (int i = 0; i < NUM_STREAMS; i++) {
    //     conv_forward_kernel_cm_and_tuning<<<dimGrid, dimBlock, 0, stream[i]>>>(device_output + (i * (output_size_stream / sizeof(float))), device_input + (i * (input_size_stream / sizeof(float))), device_mask, Batch, Map_out, Channel, Height, Width, K);
    // }
    
    conv_forward_kernel_cm_and_tuning<<<dimGrid, dimBlock>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    int output_size = Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float);

    // #pragma unroll
    // for (int i = 0; i < NUM_STREAMS; i++) {
    //     cudaMemcpyAsync(host_output + (i * (output_size_stream / sizeof(float))), device_output + (i * (output_size_stream / sizeof(float))), output_size_stream, cudaMemcpyDeviceToHost, stream[i]);
    // }
    // cudaDeviceSynchronize();

    cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_input);
    // cudaFree(device_mask);
    cudaFree(device_output);

    // #pragma unroll
    // for (int i = 0; i < NUM_STREAMS; i++) {
    //     cudaStreamDestroy(stream[i]);
    // }
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
