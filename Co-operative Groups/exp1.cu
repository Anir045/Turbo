#include<cuda_runtime.h>
#include<cooperative_groups.h>
#include<cstdio.h>
#include<stdlib.h>
namespace cg=cooperative_groups;
__global__ void test1(float *A){
    auto grid  =cg::this_grid();
    int tid = blockIdx.x*blockDim.x+threadIdx.x;

    A[tid] = blockIdx.x;
    grid.sync();
    A[tid] *= 2;
    grid.sync();
    
}

int main(){
    int device=0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("GPU: %s\n", prop.name);
    printf("SM: %d\n", prop.multiProcessorCount);

    int cooperative=0;
    cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device);

    printf("Cooperative Launch Supported: %d\n", cooperative);
    if(!cooperative){
        printf("Device does not support cooperative launch.\n");
        return 1;
    }

    constexpr int BLOCK_SIZE=512;
    int blocksPerSM=0;

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, grid_sync_test, BLOCK_SIZE,0);

    printf("Active blocks / SM: %d\n", blocksPerSM);

    int maxCooperativeBlocks = blocksPerSM*prop.multiProcessorCount;

    printf("Maximum cooperative blocks: %d\n", maxCooperativeBlocks);

    int NUM_BLOCKS = maxCooperativeBlocks;
    int N = NUM_BLOCKS*BLOCK_SIZE;

    printf("Launching %d blocks %d threads = %d threads\n", NUM_BLOCKS, BLOCK_SIZE, N);

    int *d_A = nullptr;
    cudaMalloc(&d_A, N*sizeof(int));

    void *args[] = {&d_A};

    dim3 grid(NUM_BLOCKS);
    dim3 block(BLOCK_SIZE);

    cudaError_t err = cudaLaunchCooperativeKernel( (void)*test1, grid, block, args, 0, 0);
    if(err!= cudaSuccess){
        printf("Launch Failed %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        return 1;
    }
    err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        printf(
            "Kernel failed: %s\n",
            cudaGetErrorString(err)
        );

        cudaFree(d_A);
        return 1;
    }
    int *h_A = new int[N];

    cudaMemcpy(
        h_A,
        d_A,
        N * sizeof(int),
        cudaMemcpyDeviceToHost
    );

    for (int block_id = 0;
         block_id < min(NUM_BLOCKS, 8);
         ++block_id)
    {
        int index = block_id * BLOCK_SIZE;

        printf(
            "Block %d -> A[%d] = %d\n",
            block_id,
            index,
            h_A[index]
        );
    }

    delete[] h_A;
    cudaFree(d_A);

    return 0;
}