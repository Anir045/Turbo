#define FULL_MASK 0xffffffff
#define BLOCK_DIM 256

struct SoftmaxState{
    float max;
    float sum;
};

__device__ SoftmaxState SoftmaxMerge(SoftmaxState a, SoftmaxState b){
    if(a.max == -CUDART_INF_F){
        return b;
    }
    if(b.max == -CUDART_INF_F){
        return a;
    }

    SoftmaxState result;
    result.max = fmaxf(a.max, b.max);
    float x = a.sum*(expf(a.max-result.max));
    float y = b.sum*(expf(b.max-result.max));
    result.sum = x + y;
    return result;
}

__global__ void OnlineSoftMaxInit(float *A, int N, SoftmaxState *out){
    int tid = blockIdx.x *blockDim.x +threadIdx.x;
    if(tid < N){
        out[tid].max = A[tid];
        out[tid].sum = 1.0f;
    }
}


__global__ void OnlineSoftmaxReduce(SoftmaxState *in, int N, SoftmaxState *out){
    __shared__ SoftmaxState stagingArea[(BLOCK_DIM+31)/32];
    int tx = threadIdx.x; int bx = blockIdx.x;
    int tid = bx*blockDim.x+tx;
    int lane_id = tx%32;
    int warp_id = tx/32;
    int numWarps = (BLOCK_DIM+31)/32;

    SoftmaxState x;
    x.max = -CUDART_INF_F;
    x.sum = 0.0f;
    if(tid<N){
        x = in[tid];
    }
    for(int off=16;off>0;off>>=1){
        SoftmaxState y;
        y.max = __shfl_down_sync(FULL_MASK, x.max, off);
        y.sum = __shfl_down_sync(FULL_MASK, x.sum, off);

        x = SoftmaxMerge(x,y);
    }
    if(lane_id==0){
        stagingArea[warp_id] = x;
    }
    __syncthreads();

    if(warp_id==0){
        SoftmaxState blockResult;
        if(lane_id<numWarps){
            blockResult = stagingArea[lane_id];
        }else{
            blockResult.max = -CUDART_INF_F;
            blockResult.sum = 0.0f;
        }
        for(int off1=16;off1>0;off1>>=1){
            SoftmaxState z;
            z.max = __shfl_down_sync(FULL_MASK, blockResult.max, off1);
            z.sum = __shfl_down_sync(FULL_MASK, blockResult.sum, off1);

            blockResult = SoftmaxMerge(blockResult,z);
        }

        if(lane_id==0){
            out[bx] = blockResult;
        }
    }
}

__global__ void SoftmaxNormalization(SoftmaxState *out, int N, float *B, float *A){
    int bx = blockIdx.x; int tx = threadIdx.x;
    int tid = bx*blockDim.x+tx;
    int strides = blockDim.x*gridDim.x;

    SoftmaxState finalState;
    finalState.max = out[0].max;
    finalState.sum = out[0].sum;
    float invSum = 1/finalState.sum;
    for(int i=tid;i<N;i+=2*strides){
        
        float e = expf(A[i]-finalState.max);
        B[i] = e*invSum;

        if(i+strides<N){
            float f = expf(A[i+strides]-finalState.max);
            B[i+strides] = f*invSum;
        }
    }
}

/*Running Order
1. OnlineSoftmaxInit
2. OnlineSoftmaxReduce
3. SoftmaxNormalization
*/
