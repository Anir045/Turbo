#define BLOCK_DIM 256
#define FULL_MASK 0xffffffff

__global__ void blockLevelReduction_Sum(float *A, int N, float *out){
    __shared__ float stagingArea[(BLOCK_DIM+31)/32];
    int tx = threadIdx.x; int bx = blockIdx.x;
    int tid = bx*BLOCK_DIM+tx;
    int lane_id = tx%32;
    int warp_id = tx/32;
    int numWarps = (BLOCK_DIM+31)/32;
    float blockSum = 0.0f;

    //Step 1: Get into Registers
    float x = 0.0f;
    if(tid<N){
        x = A[tid];
    }

    //Step2: Warp Reduction, 1s stage
    for(int off=16;off>0;off>>=1){
        float y = __shfl_down_sync(FULL_MASK, x, off);
        x = x+y;
    }

    //Step3: Write to shared
    if(lane_id==0){
        stagingArea[warp_id] = x;
    }
    __syncthreads();

    //Step4: Read from shared
    if(warp_id==0){
        if(lane_id<numWarps){
            blockSum = stagingArea[lane_id];
        }else{
            blockSum = 0.0f;
        }
        for(int off1=16;off1>0;off1>>=1){
            float z = __shfl_down_sync(FULL_MASK, blockSum, off1);
            blockSum += z;
        }
    }
    if(lane_id==0 && warp_id==0){
        out[bx] = blockSum;
    }
}

int main()
{
    //----------------------------------------------------------
    // Create Host Data
    //----------------------------------------------------------

    const int N = 100;

    std::vector<float> h_in(N);

    for(int i = 0; i < N; i++)
        h_in[i] = (float)(rand() % 100000);

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

        blockLevelReduction_Sum<<<numBlocks, BLOCK_DIM>>>(
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

    std::cout << "\nGPU Sum = "
              << answer
              << std::endl;

    //----------------------------------------------------------
    // CPU Verification
    //----------------------------------------------------------

    float cpu_sum=0;
    for (int i = 0; i < N; ++i) {
        cpu_sum += h_in[i];
    }

    std::cout
        << "CPU Sum = "
        << cpu_sum
        << std::endl;

    //----------------------------------------------------------
    // Cleanup
    //----------------------------------------------------------

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}