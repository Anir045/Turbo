__global__ void countPositives(float *A, int N, int res){
    __shared__ int pos[BLOCK_DIM];
    int tx = threadIdx.x; int bx = blockIdx.x;
    int tid = bx * blockDim.x + tx;
    int strides = blockDim.x*gridDim.x;
    int count = 0;
    
    for(int i=tid;i<N;i+=4*strides){
        if(i<N){
            count += (A[i]>0.0f);
        } if(i+1*strides<N){
            count += (A[i+1*strides]>0.0f);
        } if(i+2*strides<N){
            count += (A[i+2*strides]>0.0f);
        } if(i+3*strides<N){
            count += (A[i+3*strides]>0.0f);
        }
    }
    pos[tx] = count;
    __syncthreads();
    for(int j=blockDim.x/2;j>0;j/=2){
        if(tx<j){
            pos[tx] += pos[tx+j];
        }
        __syncthreads();
    }
    if(tx==0){
        atomicAdd(res, pos[0]);
    }
}