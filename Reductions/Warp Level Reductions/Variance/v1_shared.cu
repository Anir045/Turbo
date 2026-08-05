__global__ void variance(float *A, int N, float *mu, float *res){
    __shared__ float s[BLOCK_DIM];
    int tx = threadIdx.x; int bx = blockIdx.x;
    int tid = bx * blockDim.x + tx;
    int strides = blockDim.x*gridDim.x;
    float m = *mu;
    float var = 0;
    
    for(int i=tid;i<N;i+=4*strides){
        if(i<N){
            var += (A[i]-mu)*(A[i]-m);
        } if(i+1*strides<N){
            var += (A[i+1*strides]-m)*(A[i+1*strides]-m);
        } if(i+2*strides<N){
            var += (A[i+2*strides]-m)*(A[i+2*strides]-m);
        } if(i+3*strides<N){
            var += (A[i+3*strides]-m)*(A[i+3*strides]-m);
        }
    }
    s[tx] = var;
    __syncthreads();
    for(int j=blockDim.x/2;j>0;j/=2){
        if(tx<j){
            s[tx] += s[tx+j];
        }
        __syncthreads();
    }
    if(tx==0){
        atomicAdd(res, s[0]);
    }

}