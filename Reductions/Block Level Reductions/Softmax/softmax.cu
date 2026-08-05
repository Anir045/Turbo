#define FULL_MASK 0xffffffff
#define BLOCK_DIM 256

struct SoftmaxState{
    float e;
    float maxState;
    float sumState;
};

__global__ void Softmax(float *A, int N, SoftmaxState *out){
    __shared__ SoftmaxState stagingArea[(BLOCK_DIM+31)/32];
    int tx = threadIdx.x; int bx = blockIdx.x;
    int tid = bx*blockDim.x+tx;
    int lane_id = tx % 32;
    int warp_id = tx / 32;
    int numWarps = (BLOCK_DIM+31)/32;
    SoftmaxState blockResult;
    blockResult.maxState = -CUDART_INF_F;
    blockResult.sumState = 0.0f;
    SoftmaxState merge;
    merge.maxState = -CUDART_INF_F;
    merge.sumState = 0.0f;
    SoftmaxState x;
    x.maxState = -CUDART_INF_F;
    x.sumState = 0.0f;
    
    
    if(tid<N){
        x.maxState = A[tid];
        float val = A[tid];
    }
    //Calculate Max for a warp
    for(int off=16;off>0;off>>=1){
        float y = __shfl_down_sync(FULL_MASK, x.maxState, off);
        x.maxState = fmaxf(x.maxState, y);
    }
    //Broadcast max
    float warp_max = __shfl_sync(FULL_MASK, x.maxState, 0);

    //Now we we have max, for each warp
    x.e = expf(val-warp_max);
    x.sumState = x.e;

    //now we can sum
    for(int off1=16;off1>0;off1>>=1){
        float y1 = __shfl_down_sync(FULL_MASK, x.sumState, off1);
        x.sumState +=y;
    }
    //have sumState for each warp, stored in lane0 of each warp
    if(lane_id==0){
        stagingArea[warp_id] = x;
    }
    __syncthreads();
    //now we need correction factor, new max and e^(diff of maxes) * (sum)
    //ladies and gents, fasten your seatbelts!

    if(warp_id==0){
        if(lane_id<numWarps){
            blockResult = stagingArea[lane_id];
        }
        for(int off2=16;off2>0;off2>>=1){
            SoftmaxState y2;
            float y2.maxState = __shfl_down_sync(FULL_MASK, blockResult.maxState, off2);
            float y2.sumState = __shfl_down_sync(FULL_MASK, blockResult.sumState, off2);

            merge.maxState = fmaxf(blockResult.max, y2);
            merge.sumState = blockResult.sumState * expf(blockResult.maxState - merge.maxState) + y2.sumState * expf(y2.maxState - merge.maxState);
        }
    }
    if(lane_id==0){
        out[bx] = merge;
    }
    SoftmaxState finalState = out[bx];
}
    