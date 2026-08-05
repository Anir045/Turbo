#define FULL_MASK 0xffffffff
#define BLOCK_DIM 256

struct L1NormState{
    float sum;
};

__device__ L1NormState L1NormUpdate(L1NormState a, L1NormState b){
    L1NormState c;
    c.sum = a.sum +b.sum;
    return c;
}
__global__ void MakeL1State(const float *A, int N, L1NormState *in){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        in[tid].sum = fabsf(A[tid]);
    }
}
__global__ void L1Norm(L1NormState *in, int N, L1NormState *out){
    __shared__ L1NormState stagingArea[(BLOCK_DIM+31)/32];
    int bx = blockIdx.x; int tx = threadIdx.x;
    int tid = bx*blockDim.x+tx;
    int warp_id = tx/32;
    int lane_id = tx%32;
    int numWarps = (BLOCK_DIM+31)/32;
    L1NormState x;
    x.sum = 0.0f;

    if(tid<N){
        x = in[tid];
    }else{
        x.sum = 0.0f;
    }
    for(int off=16;off>0;off>>=1){
        L1NormState y;
        y.sum = __shfl_down_sync(FULL_MASK, x.sum, off);
        x = L1NormUpdate(x,y);

    }
    if(lane_id==0){
        stagingArea[warp_id] = x;
    }
    __syncthreads();
    if(warp_id==0){
        L1NormState blockResult;
        if(lane_id<numWarps){
            blockResult = stagingArea[lane_id];
        }else{
            blockResult.sum = 0.0;
        }
        for(int off1=16;off1>0;off1>>=1){
            L1NormState z;
            z.sum = __shfl_down_sync(FULL_MASK, blockResult.sum, off1);
            blockResult = L1NormUpdate(blockResult, z);
        }
        if(lane_id==0){
            out[bx] = blockResult;
        }
    }
}

#include <iostream>
#include <vector>
#include <cmath>

int main()
{
    const int N = 10;

    float h_A[N] =
    {
        1.0f,
        -2.0f,
        3.0f,
        -4.0f,
        5.0f,
        -6.0f,
        7.0f,
        -8.0f,
        9.0f,
        -10.0f
    };

    float *d_A;
    cudaMalloc(&d_A, N * sizeof(float));

    cudaMemcpy(d_A,
               h_A,
               N * sizeof(float),
               cudaMemcpyHostToDevice);

    // First state array
    L1NormState *d_state;
    cudaMalloc(&d_state, N * sizeof(L1NormState));

    MakeL1State<<<(N + BLOCK_DIM - 1)/BLOCK_DIM,
                  BLOCK_DIM>>>(d_A, N, d_state);

    cudaDeviceSynchronize();

    int currN = N;

    L1NormState *d_in = d_state;
    L1NormState *d_out;

    while(currN > 1)
    {
        int numBlocks = (currN + BLOCK_DIM - 1) / BLOCK_DIM;

        cudaMalloc(&d_out,
                   numBlocks * sizeof(L1NormState));

        L1Norm<<<numBlocks,
                 BLOCK_DIM>>>(d_in,
                              currN,
                              d_out);

        cudaDeviceSynchronize();

        if(d_in != d_state)
            cudaFree(d_in);

        d_in = d_out;
        currN = numBlocks;
    }

    L1NormState result;

    cudaMemcpy(&result,
               d_in,
               sizeof(L1NormState),
               cudaMemcpyDeviceToHost);

    std::cout << "L1 Norm = "
              << result.sum
              << std::endl;

    cudaFree(d_A);
    cudaFree(d_state);

    if(d_in != d_state)
        cudaFree(d_in);

    return 0;
}