__global__ void add(float *A, int N, float *res){
    __shared__ float a[BLOCK_DIM];
    int tx = threadIdx.x; int bx = blockIdx.x;
    int tid = bx*blockDim.x+tx;
    int strides = gridDim.x*blockDim.x;
    float sum = 0.0f;

    for(int i=tid;i<N;i+=2*strides){
        if(i<N){
            sum += A[i];
        }
        if(i+strides<N){
            sum += A[i+strides];
        }
    }
    a[tx] = sum;
    __syncthreads();

    for(int k=blockDim.x/2;k>0;k>>=1){
        if(tx<k){
            a[tx] += a[tx+k];
        }
        __syncthreads();
    }
    if(tx==0){
        atomicAdd(res, a[0]);
    }
}