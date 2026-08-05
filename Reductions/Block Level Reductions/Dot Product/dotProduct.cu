#define FULL_MASK 0xffffffff
#define BLOCK_DIM 256

__global__ blockLevelReduction_dotProduct(float *A, float *B, int N, float *out){
    __shared__ float stagingArea[(BLOCK_DIM+31)/32];
    int tx = threadIdx.x; int bx = blockIdx.x;
    int tid = bx*blockDim.x+tx;
    int lane_id = tx % 32;
    int warp_id = tx / 32;
    int numWarps = (BLOCK_DIM+31)/32;
    float blockdp = 0.0f;

    float x = 0.0f;
    if(tid<N){
        x = A[tid]*B[tid];
    }
    for(int off=16;off>0;off>>=1){
        float y = __shfl_down_sync(FULL_MASK, x, off);
        x = x+y;
    }
    if(lane_id==0){
        stagingArea[warp_id] = x;
    }
    __syncthreads();

    //Step4: Read from shared
    if(warp_id==0){
        if(lane_id<numWarps){
            blockdp = stagingArea[lane_id];
        }else{
            blockdp = 0.0f;
        }
        for(int off1=16;off1>0;off1>>=1){
            float z = __shfl_down_sync(FULL_MASK, blockdp, off1);
            blockdp += z;
        }
    }
    if(lane_id==0 && warp_id==0){
        out[bx] = blockdp;
    }
}

//cannot run this kernel again, input out into a sum kernel...

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
    const int N = 1000;

    //--------------------------------------------------
    // Host Arrays
    //--------------------------------------------------

    std::vector<float> h_A(N);
    std::vector<float> h_B(N);

    srand(time(NULL));

    for(int i=0;i<N;i++)
    {
        h_A[i] = (rand()%100)/10.0f;
        h_B[i] = (rand()%100)/10.0f;
    }

    //--------------------------------------------------
    // Device Arrays
    //--------------------------------------------------

    float *d_A;
    float *d_B;
    float *d_out;

    cudaMalloc(&d_A,N*sizeof(float));
    cudaMalloc(&d_B,N*sizeof(float));
    cudaMalloc(&d_out,N*sizeof(float));

    cudaMemcpy(d_A,
               h_A.data(),
               N*sizeof(float),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_B,
               h_B.data(),
               N*sizeof(float),
               cudaMemcpyHostToDevice);

    //--------------------------------------------------
    // FIRST PASS
    // Dot Product
    //--------------------------------------------------

    int currentN = N;

    int numBlocks =
        (currentN + BLOCK_DIM - 1)/BLOCK_DIM;

    blockLevelReduction_dotProduct<<<numBlocks,BLOCK_DIM>>>(
        d_A,
        d_B,
        currentN,
        d_out
    );

    cudaDeviceSynchronize();

    //--------------------------------------------------
    // SECOND+
    // Recursive SUM reduction
    //--------------------------------------------------

    float *d_in = d_out;

    float *d_temp;
    cudaMalloc(&d_temp,N*sizeof(float));

    currentN = numBlocks;

    while(currentN > 1)
    {
        numBlocks =
            (currentN + BLOCK_DIM - 1)/BLOCK_DIM;

        std::cout
            << "Remaining : "
            << currentN
            << " -> "
            << numBlocks
            << std::endl;

        blockLevelReduction_Sum<<<numBlocks,BLOCK_DIM>>>(
            d_in,
            currentN,
            d_temp
        );

        cudaDeviceSynchronize();

        std::swap(d_in,d_temp);

        currentN = numBlocks;
    }

    //--------------------------------------------------
    // Copy Answer
    //--------------------------------------------------

    float gpuDot;

    cudaMemcpy(&gpuDot,
               d_in,
               sizeof(float),
               cudaMemcpyDeviceToHost);

    //--------------------------------------------------
    // CPU Verification
    //--------------------------------------------------

    float cpuDot = 0.0f;

    for(int i=0;i<N;i++)
        cpuDot += h_A[i]*h_B[i];

    //--------------------------------------------------
    // Print
    //--------------------------------------------------

    std::cout
        << "\nGPU Dot = "
        << gpuDot
        << std::endl;

    std::cout
        << "CPU Dot = "
        << cpuDot
        << std::endl;

    //--------------------------------------------------
    // Cleanup
    //--------------------------------------------------

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_out);
    cudaFree(d_temp);

    return 0;
}