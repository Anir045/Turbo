__global__ void prodreduce3(float *A, int N, float *res){
    __shared__ float Ads[BLOCK_DIM];
    int tx = threadIdx.x;
    int segment = (2*blockIdx.x*blockDim.x)*CF;
    int seg_idx = segment+tx;
    float prod = 1.00;

    for(int i=0;i<CF*2;++i){
        if(seg_idx+i*BLOCK_DIM<N){
            prod = prod*A[seg_idx+i*BLOCK_DIM];
        }
    }
    Ads[tx] = prod;
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride/=2){
        __syncthreads();
        if(tx<stride){
            Ads[tx] *= Ads[tx+stride];
        }
    }
    if(tx==0){
       res[blockIdx.x] = Ads[0];
       //now plug this res back into Kernel2 or Kernel3 to reduce to 1 single number
    }
}