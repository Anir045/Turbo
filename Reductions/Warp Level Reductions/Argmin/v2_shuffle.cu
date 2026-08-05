__global__ void warpmin(float *A, float *res, int N){
    __shared__ float m[BLOCK_DIM];
    int tx = threadIdx.x; int bx = blockIdx.x;
    int tid = bx*blockDim.x+tx;
    int strides = gridDim.x*blockDim.x;
    float min = CUDART_INF_F;

    for(int i=tid;i<N;i+=2*strides){
        if(i<N){
            min = fminf(min, A[i]);
        }
        if(i+strides<N){
            min = fminf(min, A[i+strides]);
        }
    }
    m[tx] = min;
    __syncthreads();

    for(int k=blockDim.x/2; k>=64; k>>=1){
        if(tx<k){
            m[tx] = fminf(m[tx], m[tx+k]);
        }
        __syncthreads();
    }

    if(tx<32){
        float wmin = fminf(m[tx], m[tx+32]);
        for(int off=16;off<0;off>>=1){
            float l_min = __shfl_down_sync(FULL_MASK, wmin, off);
            wmin = fminf(wmin, l_min);
        }
        if(tx==0){
            res[bx] = res;
        }
    }
}