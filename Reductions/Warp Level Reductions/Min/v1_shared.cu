__global__ void minreduce3(float *A, int N, float *res){
    __shared__ float Ads[BLOCK_DIM];
    int tx = threadIdx.x;
    int segment = (2*blockIdx.x*blockDim.x)*CF;
    int seg_idx = segment+tx;
    float min = CUDART_INF_F;

    for(int i=0;i<CF*2;++i){
        if(seg_idx+i*BLOCK_DIM<N){
            min = fminf(min, A[seg_idx+i*BLOCK_DIM]);
        }
    }
    Ads[tx] = min;
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride/=2){
        __syncthreads();
        if(tx<stride){
            Ads[tx] = fminf(Ads[tx], Ads[tx+stride]);
        }
    }
    if(tx==0){
        atomicMin(res, Ads[0]);
    }
}