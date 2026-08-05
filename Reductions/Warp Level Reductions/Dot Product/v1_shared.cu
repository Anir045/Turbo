__global__ void dotproduct(float *A, float *B, int N, float *res){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int Row = 2*bx*(CF*blockDim.x)+tx;
    __shared__ float R[BLOCK_DIM];
    float sum = 0.0f;

    for(int i=0;i<2*CF;i++){
        int idx = Row+i*2*blockDim.x;
        if(idx < N){
            sum += A[idx] * B[idx];
        }
        if(idx + BLOCK_DIM < N){
            sum += A[idx + BLOCK_DIM] * B[idx + BLOCK_DIM];
        }
    }

    R[tx] = sum;
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride/=2){
        if(tx<stride){
            R[tx] += R[tx+stride];
        }
        __syncthreads();
    }
    if(tx==0){
        atomicAdd(res, R[0]);
    }

}