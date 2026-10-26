#include<cuda_runtime.h>
#include<cooperative_groups.h>
#include<cstdio.h>
#include<stdlib.h>
namespace cg=cooperative_groups;

__global__ void test2(float *A){
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_parition<32>(block);
    printf("BLOCK: %d TILE: %d\n", block, tile);
    block.sync();
}

int main(){
    int device = 0;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("GPU: %s\n", prop.name);
    printf("SMs: %d\n", prop.multiProcessorCount);

    // ------------------------------------------------
    // 1. Check cooperative launch support
    // ------------------------------------------------

    int cooperative = 0;

    cudaDeviceGetAttribute(
        &cooperative,
        cudaDevAttrCooperativeLaunch,
        device
    );

    printf(
        "Cooperative launch supported: %d\n",
        cooperative
    );

    if (!cooperative) {
        printf("Device does not support cooperative launch.\n");
        return 1;
    }

    // ------------------------------------------------
    // 2. Choose block size
    // ------------------------------------------------

    constexpr int BLOCK_SIZE = 256;

    // ------------------------------------------------
    // 3. Ask CUDA how many of OUR blocks
    //    can simultaneously reside on one SM.
    // ------------------------------------------------

    int blocksPerSM = 0;

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocksPerSM,
        grid_sync_test,
        BLOCK_SIZE,
        0
    );

    printf(
        "Active blocks / SM: %d\n",
        blocksPerSM
    );

    // ------------------------------------------------
    // 4. Maximum cooperative grid
    // ------------------------------------------------

    int maxCooperativeBlocks =
        blocksPerSM * prop.multiProcessorCount;

    printf(
        "Maximum cooperative blocks: %d\n",
        maxCooperativeBlocks
    );

    // We'll deliberately use something smaller.
    int NUM_BLOCKS = maxCooperativeBlocks;

    int N = NUM_BLOCKS * BLOCK_SIZE;

    printf(
        "Launching %d blocks × %d threads = %d threads\n",
        NUM_BLOCKS,
        BLOCK_SIZE,
        N
    );

    // ------------------------------------------------
    // 5. Allocate
    // ------------------------------------------------

    int *d_A = nullptr;

    cudaMalloc(
        &d_A,
        N * sizeof(int)
    );

    // ------------------------------------------------
    // 6. Prepare kernel arguments
    // ------------------------------------------------

    void *args[] = {
        &d_A
    };

    // ------------------------------------------------
    // 7. COOPERATIVE LAUNCH
    // ------------------------------------------------

    dim3 grid(NUM_BLOCKS);
    dim3 block(BLOCK_SIZE);

    cudaError_t err =
        cudaLaunchCooperativeKernel(
            (void*)grid_sync_test,
            grid,
            block,
            args,
            0,
            0
        );

    if (err != cudaSuccess) {
        printf(
            "Cooperative launch failed: %s\n",
            cudaGetErrorString(err)
        );

        cudaFree(d_A);
        return 1;
    }

    // ------------------------------------------------
    // 8. Wait for kernel completion
    // ------------------------------------------------

    err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        printf(
            "Kernel failed: %s\n",
            cudaGetErrorString(err)
        );

        cudaFree(d_A);
        return 1;
    }

    // ------------------------------------------------
    // 9. Copy back
    // ------------------------------------------------

    int *h_A = new int[N];

    cudaMemcpy(
        h_A,
        d_A,
        N * sizeof(int),
        cudaMemcpyDeviceToHost
    );

    // ------------------------------------------------
    // 10. Inspect a few values
    // ------------------------------------------------

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