__global__ void warp_product(float *A, int N, float *res){
    __shared__ float p[BLOCK_DIM];
    int tx = threadIdx.x; int bx = blockIdx.x;
    int tid = bx*blockDim.x+tx;
    int strides = gridDim.x*blockDim.x;
    float prod = 1.0f;

    for(int i=tid;i<N;i+=2*strides){
        if(i<N){
            prod *= A[i];
        }
        if(i+strides<N){  
            prod *= A[i+strides];
        }
    }
    p[tx] = prod;
    __syncthreads();

    for(int k=blockDim.x/2;k>=64;k>>=1){
        if(tx<k){
            p[tx] *= p[tx+k];
        }
        __syncthreads();
    }

    if(tx<32){
        float p_final = p[tx] * p[tx+32];
        for(int off=16;off>0;off>>=1){
            p_final *= __shfl_down_sync(FULL_MASK, p_final, off);
        }
        if(tx==0){
            res[bx] = p_final;
        }
    }

}