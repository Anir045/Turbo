//todo: decouple transform and reduction steps

#define FULL_MASK 0xffffffff

struct maxPair{
    float value;
    int index;
};
__device__ inline maxPair helper(maxPair a, maxPair b){
    if(a.value<b.value){
        return b;
    } 
    if(a.value > b.value){
        return a;
    }
    if(a.index < b.index){
        return a;
    }
    return b;
}

__global__ void argmax(const float *A, int N, float *res){
    __shared__ maxPair a[BLOCK_DIM];
    int tx = threadIdx.x; int bx = blockIdx.x;
    int tid = bx*blockDim.x+tx;
    int stride = blockDim.x*gridDim.x;
    maxPair current;
    current.value = -CUDART_INF_F;
    current.index = -1;


    for(int i=tid;i<N;i+=2*stride){
        if(current.value<A[i] || (current.value==A[i] && i<current.index)){
            current.value = A[i];
            current.index = i;
        }
        if(i+stride<N){
            if(current.value<A[i+stride] || (current.value == A[i+stride] && i+stride<current.index)){
                current.value=A[i+stride];
                current.index=i+stride;
            }
        }
    }
    a[tx] = current;
    __syncthreads();

    for(int k=blockDim.x/2;k>=64;k>>=1){
        if(tx<k){
            a[tx] = helper(a[tx], a[tx+k]);
        }
        __syncthreads();
    }

    if(tx<32){
        maxPair val;
        if(a[tx].value<a[tx+32].value){
            val.index = a[tx+32].index;
            val.value = a[tx+32].value;
        }else{
            val.index = a[tx].index;
            val.value = a[tx+32].value;
        }
        for(int off=16;off>0;off>>=1){
            maxPair local;
            local.value = __shfl_down_sync(FULL_MASK, val.value, off);
            local.index = __shfl_down_sync(FULL_MASK, val.index, off);
            val = helper(val, local);
        }
        if(tx==0){
            res[bx] = val;
        }
    }
}