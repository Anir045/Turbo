__global__ void blockLevelReduction(float *A, int N, float *out){
    __shared__ float stagingArea[(BLOCK_DIM+31)/32];
    int tx = threadIdx.x; int bx = blockIdx.x;
    int warp_id = tx / 32;
    int lane_id = tx % 32;
    int numWarps = (BLOCK_DIM+31)/32;
    int tid = bx*blockDim.x+tx;
    float max = -CUDART_INF_F;
    float blockMax = -CUDART_INF_F;

    //Stage 1: Get into registers:
    float x = A[tid];

    //Stage 2: Reduction Stage1-> Multiple Warps on a block
    for(int off=16;off>0;off>>=1){
        float y = __shfl_down_sync(0xffffffff, x, off);
        x = fmaxf(x, y);
    }
    
    //Stage 3: Copy back to staging Area, intermediate results
    if(lane_id==0){
        stagingArea[warp_id] = x;
    }
    __syncthreads();

    //Stage 4: Read from staging Area, prepare for final reduction
    if(warp_id==0){
        if(lane_id<numWarps){
            blockMax = stagingArea[lane_id];
        }
        
        //Stage 5: Final Reduction
        for(int off1=16;off1>0;off1>>=1){
            float z = __shfl_down_sync(0xffffffff, blockMax, off1);
            blockMax = fmaxf(blockMax, z);
        }
    }

    //Stage 6: write to output, indexed by blockIdx
    if(lane_id==0 && warp_id==0){
        out[bx] = blockMax;
    }

}


int main()
{
    //----------------------------------------------------------
    // Create Host Data
    //----------------------------------------------------------

    const int N = 2000000;

    std::vector<float> h_in(N);

    for(int i = 0; i < N; i++)
        h_in[i] = (float)(rand() % 100000);

    // Put an obvious maximum somewhere
    //h_in[543210] = 9999999.0f;

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

        blockLevelReduction<<<numBlocks, BLOCK_DIM>>>(
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

    std::cout << "\nGPU Maximum = "
              << answer
              << std::endl;

    //----------------------------------------------------------
    // CPU Verification
    //----------------------------------------------------------

    float cpuMax = *std::max_element(
        h_in.begin(),
        h_in.end()
    );

    std::cout
        << "CPU Maximum = "
        << cpuMax
        << std::endl;

    //----------------------------------------------------------
    // Cleanup
    //----------------------------------------------------------

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}