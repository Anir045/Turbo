__global__ void l1Norm(float*A, int N, float *res){
    __shared__ float l1[BLOCK_DIM];
    int bx = blockIdx.x; int tx = threadIdx.x;
    int start_block = bx * (2* CF * blockDim.x);
    int start_tx = start_block+tx;
    float sum=0.0f;

    for(int i=0;i<CF;++i){
        int idx = start_tx + (i*2*BLOCK_DIM);
        if(idx<N){
            sum += fabsf(A[idx]);
        }
        if(idx+BLOCK_DIM<N){
            sum += fabsf(A[idx+BLOCK_DIM]);
        }
    }
    l1[tx] = sum;
    __syncthreads();

    for(int stride = blockDim.x/2; stride>0; stride/=2){
        if(tx<stride){
            l1[tx] += l1[stride+tx]; 
        }
        __syncthreads();
    }
    if(tx==0){
        atomicAdd(res, l1[0]);
    }
    
}