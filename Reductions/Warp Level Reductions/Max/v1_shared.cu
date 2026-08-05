__global__ void maxReduce(float *A, int N, float *res){
    __shared__ float Ads[BLOCK_DIM];
    int segment = (2*blockIdx.x*blockDim.x)*CF;
    int tx = threadIdx.x;
    int seg_idx = segment+tx;
    float max = -CUDART_INF_F;

    //coarsening loop for 1st stage
    for(int i=0;i<CF*2;i++){
        if(seg_idx+i*BLOCK_DIM<N){
            max = fmaxf(max, Ads[seg_idx+i*BLOCK_DIM]);
        }
    }
    Ads[tx] = max;
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride/=2){
        __syncthreads();
        if(tx<stride){
            Ads[tx] = fmaxf(Ads[tx], Ads[tx+stride]);
        }
    }
    if(tx==0){
        atomicMax((int*)res, __float_as_int(Ads[0]));
    }

}