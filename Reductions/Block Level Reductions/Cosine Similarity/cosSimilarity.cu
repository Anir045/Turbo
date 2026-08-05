#define FULL_MASK 0xffffffff
#define BLOCK_DIM 256

struct CosineState{
    float dotProduct;
    float L2A;
    float L2B;
};
__global__ void blockLevelReduction_cosSim(float *A, float *B, int N, CosineState *out){
    __shared__ CosineState stagingArea[(BLOCK_DIM+31)/32];
    int tx = threadIdx.x; int bx = blockIdx.x;
    int tid = bx*blockDim.x+tx;
    int lane_id = tx % 32;
    int warp_id = tx / 32;
    int numWarps = (BLOCK_DIM+31)/32;
    CosineState blockResult;
    blockResult.dotProduct = 0.0f;
    blockResult.L2A = 0.0f;
    blockResult.L2B = 0.0f;
    CosineState x;
    x.dotProduct = 0.0f;
    x.L2A = 0.0f;
    x.L2B = 0.0f;

    if(tid<N){
        float a = A[tid]; float b = B[tid];
        x.dotProduct = a*b;
        x.L2A = a*a;
        x.L2B = b*b;
    }
    for(int off=16;off>0;off>>=1){
        float y_dp = __shfl_down_sync(FULL_MASK, x.dotProduct, off);
        float y_l2a = __shfl_down_sync(FULL_MASK, x.L2A, off);
        float y_l2b = __shfl_down_sync(FULL_MASK, x.L2B, off);

        x.dotProduct += y_dp;
        x.L2A += y_l2a;
        x.L2B += y_l2b;
    }
    if(lane_id==0){
        stagingArea[warp_id] = x;
    }
    __syncthreads();

    if(warp_id==0){
        if(lane_id<numWarps){
            blockResult = stagingArea[lane_id];
        }else{
            blockResult = {0.0f, 0.0f, 0.0f};
        }
        for(int off1=16;off1>0;off1>>=1){
            float z_dp = __shfl_down_sync(FULL_MASK, blockResult.dotProduct, off1);
            float z_l2a = __shfl_down_sync(FULL_MASK, blockResult.L2A, off1);
            float z_l2b = __shfl_down_sync(FULL_MASK, blockResult.L2B, off1);

            blockResult.dotProduct += z_dp;
            blockResult.L2A += z_l2a;
            blockResult.L2B += z_l2b;
        }
    }
    if(lane_id==0 && warp_id==0){
        out[bx] = blockResult;
    }
}

__global__ void blockLevelReduction_cosSimSum(CosineState *A, int N, CosineState *out){
    __shared__ CosineState stagingArea[(BLOCK_DIM+31)/32];
    int tx = threadIdx.x; int bx = blockIdx.x;
    int tid = bx*blockDim.x+tx;
    int lane_id = tx % 32;
    int warp_id = tx / 32;
    int numWarps = (BLOCK_DIM+31)/32;
    CosineState blockResult;
    blockResult.dotProduct = 0.0f;
    blockResult.L2A = 0.0f;
    blockResult.L2B = 0.0f;
    CosineState x;
    x.dotProduct = 0.0f;
    x.L2A = 0.0f;
    x.L2B = 0.0f;

    if(tid<N){
        x = A[tid];
    }
    for(int off=16;off>0;off>>=1){
        float y_dp = __shfl_down_sync(FULL_MASK, x.dotProduct, off);
        float y_l2a = __shfl_down_sync(FULL_MASK, x.L2A, off);
        float y_l2b = __shfl_down_sync(FULL_MASK, x.L2B, off);

        x.dotProduct += y_dp;
        x.L2A += y_l2a;
        x.L2B += y_l2b;
    }
    if(lane_id==0){
        stagingArea[warp_id] = x;
    }
    __syncthreads();

    if(warp_id==0){
        if(lane_id<numWarps){
            blockResult = stagingArea[lane_id];
        }else{
            blockResult = {0.0f, 0.0f, 0.0f};
        }
        for(int off1=16;off1>0;off1>>=1){
            float z_dp = __shfl_down_sync(FULL_MASK, blockResult.dotProduct, off1);
            float z_l2a = __shfl_down_sync(FULL_MASK, blockResult.L2A, off1);
            float z_l2b = __shfl_down_sync(FULL_MASK, blockResult.L2B, off1);

            blockResult.dotProduct += z_dp;
            blockResult.L2A += z_l2a;
            blockResult.L2B += z_l2b;
        }
    }
    if(lane_id==0 && warp_id==0){
        out[bx] = blockResult;
    }

}