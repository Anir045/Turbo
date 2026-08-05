__global__ void l2Norm(float*A, int N, float *res){
    __shared__ float l2[BLOCK_DIM];
    int bx = blockIdx.x; int tx = threadIdx.x;
    int start_block = bx* (2*CF*blockDim.x);
    int start_tx = start_block+tx;
    float sum=0.0f;

    for(int i=0;i<2*CF;++i){
        int idx = start_tx+(i*2*BLOCK_DIM);
        if(idx<N){
            sum += A[idx] * A[idx];
        }
        if(idx+BLOCK_DIM<N){
            sum += A[idx+BLOCK_DIM] * A[idx+BLOCK_DIM];
        }
    }
    l2[tx] = sum;
    __syncthreads();

    for(int stride = blockDim.x/2; stride>0; stride/=2){
        if(tx<stride){
            l2[tx] += l2[stride+tx]; 
        }
        __syncthreads();
    }
    if(tx==0){
        atomicAdd(res, l2[tx]);
    }
    //float l2_norm = sqrtf(h_res); after copying it to host code in main() 
}