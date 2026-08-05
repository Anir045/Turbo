__global__ void warpmax(float *A, int N, float *res){
    __shared__ float m[BLOCK_DIM];
    int tx = threadIdx.x; int bx = blockIdx.x;
    int tid = bx*blockDim.x+tx;
    int strides = gridDim.x*blockDim.x;
    float max = -CUDART_INF_F;

    for(int i=tid;i<N;i+=2*strides){
        if(i<N){
            max = fmaxf(max, A[i]);
        }
        if(i+strides<N){
            max = fmaxf(max, A[i+strides]);
        }
    }
    m[tx] = max;
    __syncthreads();

    for(int k=blockDim.x/2;k>=64;k>>=1){
        if(tx<k){
            m[tx] = fmaxf(m[tx], m[tx+k]);
        }
        __syncthreads();
    }
    if(tx<32){
        float wmax = fmaxf(m[tx], m[tx+32]);
        for(int off=16; off>0; off>>=1){
            float l_max = __shfl_down_sync(FULL_MASK, wmax, off);
            wmax = fmaxf(wmax, l_max);
        }
        if(tx==0){
        res[bx] = wmax;
        }
    }
}