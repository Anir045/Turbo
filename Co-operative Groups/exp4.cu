#include<cuda_runtime.h>
#include<cooperative_groups.h>
#include<cstdio>
namespace cg = cooperative_groups;
constexpr int BLOCK_DIM = 256;
constexpr int NUM_BLOCKS = 160;
constexpr int N = BLOCK_DIM * NUM_BLOCKS;

__global__ void PrefixSum(const float* A, float* d_global_scan, float* B){
    cg::thread_block block = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();

    int tid = grid.thread_rank();
    int tx = block.thread_rank();
    int bx = block.group_index().x;

    extern __shared__ float partialSums[];

    partialSums[tx] = A[tid];
    block.sync();

    //begin local Brent Kung

    for(int up=1;up<BLOCK_DIM;up<<=1){
        int index = (tx+1)*2*up-1;
        if(index<BLOCK_DIM){
            partialSums[index] += partialSums[index-up];
        }
        block.sync();
    }

    for(int down=BLOCK_DIM/4; down>0;down>>=1){
        int index = (tx+1)*2*down-1+down;
        if(index<BLOCK_DIM){
            partialSums[index] += partialSums[index-down];
        }
        block.sync();
    }
    

    //at this moment, all blocks have their own partial inclusive sums

    float local_prefix = partialSums[tx];

    //let thread0 of each block work now
    if(tx==0){
        d_global_scan[bx] = partialSums[BLOCK_DIM-1];
    }
    grid.sync();

    if(bx==0){
        if(tx<NUM_BLOCKS){
            partialSums[tx] = d_global_scan[tx];
        }else{
            partialSums[tx] = 0.0f;
        }
        block.sync();

        //begin 2nd level brent kung

        for(int up=1;up<BLOCK_DIM;up<<=1){
            int index = (tx+1)*2*up-1;
            if(index<BLOCK_DIM){
                partialSums[index] += partialSums[index-up];
            }
            block.sync();
        }

        for(int down=BLOCK_DIM/4;down>0;down>>=1){
            int index = (tx+1)*2*down-1+down;
            if(index<BLOCK_DIM){
                partialSums[index] += partialSums[index-down];
            }
            block.sync();
        }

        if(tx<NUM_BLOCKS){
            d_global_scan[tx] = partialSums[tx];
        }
    }
    grid.sync();
    
    float block_offset = 0.0f;
    if(bx==0){
        block_offset==0.0f;
    }else{
        block_offset = d_global_scan[bx-1];
    }

    B[tid] = local_prefix + block_offset;
}

__global__ void PrefixSumInf(const float *A, float *d_global_scan, float *B){
    cg::thread_block block = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();

    int tx = block.thread_rank();
    int bx = block.grid_index().x;
    int grid_size = grid.size();
    float carry_on = 0.0f;

    for(int strides = 0; strides<N; strides+=grid_size){
        
    }
}




int main()
{
    int device = 0;

    // --------------------------------------------------
    // 1. Select device
    // --------------------------------------------------

    cudaSetDevice(device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("GPU: %s\n", prop.name);
    printf("SMs: %d\n", prop.multiProcessorCount);

    // --------------------------------------------------
    // 2. Check cooperative launch support
    // --------------------------------------------------

    int cooperative = 0;

    cudaDeviceGetAttribute(
        &cooperative,
        cudaDevAttrCooperativeLaunch,
        device
    );

    if (!cooperative) {
        printf("Cooperative launch NOT supported.\n");
        return 1;
    }

    printf("Cooperative launch: YES\n");

    // --------------------------------------------------
    // 3. Ask CUDA how many blocks/SM THIS kernel allows
    // --------------------------------------------------

    int blocksPerSM = 0;

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocksPerSM,
        PrefixSum,
        BLOCK_DIM,
        BLOCK_DIM * sizeof(float)
    );

    printf("Active blocks / SM: %d\n", blocksPerSM);

    int maxCooperativeBlocks =
        blocksPerSM * prop.multiProcessorCount;

    printf(
        "Maximum cooperative blocks: %d\n",
        maxCooperativeBlocks
    );

    // --------------------------------------------------
    // 4. Make sure our requested grid fits
    // --------------------------------------------------

    if (NUM_BLOCKS > maxCooperativeBlocks) {

        printf(
            "ERROR: requested %d blocks, "
            "but only %d can reside cooperatively.\n",
            NUM_BLOCKS,
            maxCooperativeBlocks
        );

        return 1;
    }

    // --------------------------------------------------
    // 5. Allocate device memory
    // --------------------------------------------------

    float* d_A = nullptr;
    float* d_scan = nullptr;
    float* d_B = nullptr;

    cudaMalloc(
        &d_A,
        N * sizeof(float)
    );

    cudaMalloc(
        &d_scan,
        NUM_BLOCKS * sizeof(float)
    );

    cudaMalloc(
        &d_B,
        N * sizeof(float)
    );

    // --------------------------------------------------
    // 6. Initialize input
    // --------------------------------------------------

    float* h_A = new float[N];
    float* h_B = new float[N];

    for (int i = 0; i < N; ++i)
        h_A[i] = 1.0f;

    cudaMemcpy(
        d_A,
        h_A,
        N * sizeof(float),
        cudaMemcpyHostToDevice
    );

    // --------------------------------------------------
    // 7. Kernel arguments
    // --------------------------------------------------

    void* args[] = {
        &d_A,
        &d_scan,
        &d_B
    };

    dim3 grid(NUM_BLOCKS);
    dim3 block(BLOCK_DIM);

    // --------------------------------------------------
    // 8. THE COOPERATIVE LAUNCH
    // --------------------------------------------------

    cudaError_t err =
        cudaLaunchCooperativeKernel(
            (void*)PrefixSum,
            grid,
            block,
            args,
            BLOCK_DIM * sizeof(float),
            nullptr
        );

    if (err != cudaSuccess) {

        printf(
            "Cooperative launch failed: %s\n",
            cudaGetErrorString(err)
        );

        return 1;
    }

    // --------------------------------------------------
    // 9. Wait for kernel
    // --------------------------------------------------

    err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {

        printf(
            "Kernel execution failed: %s\n",
            cudaGetErrorString(err)
        );

        return 1;
    }

    // --------------------------------------------------
    // 10. Copy result back
    // --------------------------------------------------

    cudaMemcpy(
        h_B,
        d_B,
        N * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    // --------------------------------------------------
    // 11. Inspect
    // --------------------------------------------------

    printf("\nFirst 10 results:\n");

    for (int i = 0; i < 10; ++i)
        printf(
            "B[%d] = %.1f\n",
            i,
            h_B[i]
        );

    printf("\nLast 5 results:\n");

    for (int i = N - 5; i < N; ++i)
        printf(
            "B[%d] = %.1f\n",
            i,
            h_B[i]
        );

    // Expected for all-ones input:
    //
    // B[i] = i + 1

    // --------------------------------------------------
    // 12. Cleanup
    // --------------------------------------------------

    cudaFree(d_A);
    cudaFree(d_scan);
    cudaFree(d_B);

    delete[] h_A;
    delete[] h_B;

    return 0;
}