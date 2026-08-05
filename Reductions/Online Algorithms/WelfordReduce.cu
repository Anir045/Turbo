#define BLOCK_DIM 256
#define FULL_MASK 0xffffffff

struct RunningNorm{
    float nums;
    float mean;
    float M2;
};

__device__ RunningNorm merge(RunningNorm a, RunningNorm b){
    //correction factors
    if(a.nums == 0){
        return b;
    }
    if(b.nums == 0){
        return a;
    }
    RunningNorm result;

    result.nums = a.nums + b.nums;
    float delta = b.mean - a.mean;
    result.mean = a.mean + delta*(b.nums/result.nums);
    result.M2 = a.M2 + b.M2 + delta*delta*((a.nums*b.nums)/result.nums);

    return result;
}
__global__ void WelfordInit(float *A, int N, RunningNorm *out){
    int tid = blockIdx.x *blockDim.x +threadIdx.x;

    if(tid < N){
        out[tid].nums = 1.0f;
        out[tid].mean = A[tid];
        out[tid].M2 = 0.0f;
    }
}

__global__ void WelfordReduce(RunningNorm *in, int N, RunningNorm *out){
    __shared__ RunningNorm stagingArea[(BLOCK_DIM+31)/32];
    int tx = threadIdx.x; int bx = blockIdx.x;
    int tid = bx*blockDim.x+tx;
    int lane_id = tx%32;
    int warp_id = tx/32;
    int numWarps = (BLOCK_DIM+31)/32;

    RunningNorm x;
    x.nums = 0;
    x.mean = 0;
    x.M2 = 0;
    if(tid<N){
        x.nums = 1.0f;
        x.mean = A[tid];
        x.M2 = 0.0f;
    }
    for(int off=16;off>0;off>>=1){
        RunningNorm y;
        y.nums = __shfl_down_sync(FULL_MASK, x.nums, off);
        y.mean = __shfl_down_sync(FULL_MASK, x.mean, off);
        y.M2 = __shfl_down_sync(FULL_MASK, x.M2, off);
        x = merge(x, y);
    }
    if (lane_id==0) {
        stagingArea[warp_id] = x;
    }
    __syncthreads();
    if(warp_id==0){
        RunningNorm blockResult;
        if(lane_id<numWarps){
            blockResult = stagingArea[lane_id];
        }else{
            blockResult.nums = 0.0f;
            blockResult.mean = 0.0f;
            blockResult.M2 = 0.0f;
        }
        for(int off1=16;off1>0;off1>>=1){
            RunningNorm z;
            z.nums = __shfl_down_sync(FULL_MASK, blockResult.nums, off1);
            z.mean = __shfl_down_sync(FULL_MASK, blockResult.mean, off1);
            z.M2 = __shfl_down_sync(FULL_MASK, blockResult.M2, off1);

            blockResult = merge(blockResult, z);
        }
        if(lane_id==0){
            out[bx] = blockResult;
        }
    }
}