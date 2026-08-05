__global__ void RMSnorm(float *A, int M, int N, float *res){
    __shared__ float rms[BLOCK_DIM];
    int tx = threadIdx.x; int bx = blockIdx.x;
    int tid = bx *blockDim.x + tx;
    int strides = blockDim.x * gridDim.x;
    float sum = 0.0f;

    for(int i=tid;i<M*N;i+= 2*strides){
        if(i<M*N){
            sum += A[i] * A[i];
        }
        if(i+strides<M*N){
            sum += A[i+strides] * A[i+strides];
        }
    }
    rms[tx] = sum;
    __syncthreads();

    for(int j=blockDim.x/2;j>0;j/=2){
        if(tx<j){
            rms[tx] += rms[tx+j];
        }
        __syncthreads();
    }
    if(tx==0){
        atomicAdd(res, rms[0]);
    }
}