#define FULL_MASK 0xffffffff
#define BLOCK_DIM 256

struct StatsState
{
    float sum;
    float maxVal;
    float minVal;
    int positives;
    int negatives;
    float products;
};

__device__ StatsState StatsUpdate(StatsState a, StatsState b){
    StatsState c;
    c.sum = a.sum+b.sum;
    c.maxVal = fmaxf(a.maxVal, b.maxVal);
    c.minVal = fminf(a.minVal, b.minVal);
    c.positives = a.positives + b.positives;
    c.negatives = a.negatives + b.negatives;
    c.products = a.products*b.products;
    return c;
}
__global__ void StatsInit(float *A, int N, StatsState *out){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N){
        out[tid].sum = 0.0f;
        out[tid].maxVal = -CUDART_INF_F;
        out[tid].minVal = CUDART_INF_F;
        out[tid].positives = 0;
        out[tid].negatives = 0;
        out[tid].products = 1.0f;
    }
}
__global__ void StatsReduce(const float *A, int N, StatsState *out){
    __shared__ StatsState stagingArea[(BLOCK_DIM+31)/32];
    int tx = threadIdx.x; int bx = blockIdx.x;
    int tid = bx*blockDim.x+tx;
    int lane_id = tx%32;
    int warp_id = tx/32;
    int numWarps = (BLOCK_DIM+31)/32;

    StatsState x;
    x.sum = 0.0f;
    x.maxVal = -CUDART_INF_F;
    x.minVal = CUDART_INF_F;
    x.positives = 0;
    x.negatives = 0;
    x.products = 1.0f;

    if(tid<N){
        float val = A[tid];
        x.sum = val;
        x.maxVal = val;
        x.minVal = val;
        x.products = val;
        if(val>0.0f){
            x.positives = 1;
            x.negatives = 0;
        }
        if(val<0.0f){
            x.negatives = 1;
            x.positives = 0;
        }
    }
    for(int off=16;off>0;off>>=1){
        StatsState y;
        y.sum = __shfl_down_sync(FULL_MASK, x.sum, off);
        y.maxVal = __shfl_down_sync(FULL_MASK, x.maxVal, off);
        y.minVal = __shfl_down_sync(FULL_MASK, x.minVal, off);
        y.positives = __shfl_down_sync(FULL_MASK, x.positives, off);
        y.negatives = __shfl_down_sync(FULL_MASK, x.negatives, off);
        y.products = __shfl_down_sync(FULL_MASK, x.products, off);
        x = StatsUpdate(x,y);
    }
    if(lane_id==0){
        stagingArea[warp_id] = x;
    }
    __syncthreads();

    if(warp_id==0){
        StatsState blockResult;
        if(lane_id<numWarps){
            blockResult = stagingArea[lane_id];
        }else{
            blockResult = {0.0f, -CUDART_INF_F, CUDART_INF_F, 0, 0, 1.0f};
        }
        for(int off1=16;off1>0;off1>>=1){
            StatsState z;
            z.sum = __shfl_down_sync(FULL_MASK, blockResult.sum, off1);
            z.maxVal = __shfl_down_sync(FULL_MASK, blockResult.maxVal, off1);
            z.minVal = __shfl_down_sync(FULL_MASK, blockResult.minVal, off1);
            z.positives = __shfl_down_sync(FULL_MASK, blockResult.positives, off1);
            z.negatives = __shfl_down_sync(FULL_MASK, blockResult.negatives, off1);
            z.products = __shfl_down_sync(FULL_MASK, blockResult.products, off1);
            blockResult = StatsUpdate(blockResult, z);
        }
        if(lane_id==0){
            out[bx] = blockResult;
        }
    }

}