#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <cstdlib>
#include <cmath>

#define FULL_MASK 0xffffffff
#define BLOCK_DIM 256

__global__ void blockLevelReduction_Product(float *A, int N, float *out){
    __shared__ float stagingArea[(BLOCK_DIM+31)/32];
    int tx = threadIdx.x; int bx = blockIdx.x;
    int tid = bx*blockDim.x+tx;
    int lane_id = tx % 32;
    int warp_id = tx / 32;
    int numWarps = (BLOCK_DIM+31)/32;
    float blockProd = 1.0f;

    float x = 1.0f;
    if(tid<N){
        x = A[tid];
    }
    for(int off=16;off>0;off>>=1){
        float y = __shfl_down_sync(FULL_MASK, x, off);
        x *=y;
    }
    if(lane_id==0){
        stagingArea[warp_id] = x;
    }
    if(warp_id==0){
        if(lane_id<numWarps){
            blockProd = stagingArea[lane_id];
        }else{
            blockProd = 1.0;
        }

        for(int off1=16, off1>0;off>>=1){
            float z = __shfl_down_sync(FULL_MASK, blockProd, off);
            blockProd *=z;
        }
    }
    if(lane_id==0 && warp_id==0){
        out[bx] = blockProd;
    }
}

int main()
{
    //----------------------------------------------------------
    // Create Host Data
    //----------------------------------------------------------

    const int N = 100;
    

   std::vector<float> h_in = {
    1.2f, 0.8f, 1.5f, 2.1f, 0.5f, 1.1f, 1.7f, 0.9f, 1.3f, 2.2f,
    0.6f, 1.4f, 1.8f, 0.7f, 1.2f, 2.5f, 0.4f, 1.6f, 1.1f, 1.9f,
    0.9f, 1.3f, 2.0f, 0.5f, 1.2f, 1.7f, 0.8f, 1.4f, 2.3f, 0.6f,
    1.1f, 1.5f, 1.8f, 0.7f, 1.3f, 2.4f, 0.5f, 1.6f, 1.2f, 1.9f,
    0.8f, 1.4f, 2.1f, 0.6f, 1.3f, 1.7f, 0.9f, 1.5f, 2.2f, 0.7f,
    1.2f, 0.8f, 1.5f, 2.1f, 0.5f, 1.1f, 1.7f, 0.9f, 1.3f, 2.2f,
    0.6f, 1.4f, 1.8f, 0.7f, 1.2f, 2.5f, 0.4f, 1.6f, 1.1f, 1.9f,
    0.9f, 1.3f, 2.0f, 0.5f, 1.2f, 1.7f, 0.8f, 1.4f, 2.3f, 0.6f,
    1.1f, 1.5f, 1.8f, 0.7f, 1.3f, 2.4f, 0.5f, 1.6f, 1.2f, 1.9f,
    0.8f, 1.4f, 2.1f, 0.6f, 1.3f, 1.7f, 0.9f, 1.5f, 2.2f, 0.7f
};

    // Put an obvious maximum somewhere

    //----------------------------------------------------------
    // Allocate GPU Memory
    //----------------------------------------------------------

    float *d_in;
    float *d_out;

    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));

    cudaMemcpy(
        d_in,
        h_in.data(),
        N * sizeof(float),
        cudaMemcpyHostToDevice
    );

    //----------------------------------------------------------
    // Recursive Reduction
    //----------------------------------------------------------

    int currentN = N;
    int pass = 1;

    while(currentN > 1)
    {
        int numBlocks =
            (currentN + BLOCK_DIM - 1) / BLOCK_DIM;

        std::cout
            << "Pass "
            << pass++
            << " : "
            << currentN
            << " -> "
            << numBlocks
            << std::endl;

        blockLevelReduction_Product<<<numBlocks, BLOCK_DIM>>>(
            d_in,
            currentN,
            d_out
        );

        cudaDeviceSynchronize();

        std::swap(d_in, d_out);

        currentN = numBlocks;
    }

    //----------------------------------------------------------
    // Copy Final Maximum Back
    //----------------------------------------------------------

    float answer;

    cudaMemcpy(
        &answer,
        d_in,
        sizeof(float),
        cudaMemcpyDeviceToHost
    );

    std::cout << "\nGPU Prod = "
              << answer
              << std::endl;

    //----------------------------------------------------------
    // CPU Verification
    //----------------------------------------------------------

    float cpu_sum=1;
    for (int i = 0; i < N; ++i) {
        cpu_sum *= h_in[i];
    }

    std::cout
        << "CPU Prod = "
        << cpu_sum
        << std::endl;

    //----------------------------------------------------------
    // Cleanup
    //----------------------------------------------------------

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}