#define BLOCK_DIM 256

__global__ void Blelloch(float *A, int N, float *B){
    __shared__ float partialSums[BLOCK_DIM];
    int tx = threadIdx.x; int bx = blockIdx.x;
    int tid = bx*blockDim.x+tx;

    if(tid<N){
        partialSums[tx] = A[tid];
    }else{
        partialSums[tx] = 0.0f;
    }
    __syncthreads();

    //upsweep reduce
    for(int up=1;up<BLOCK_DIM;up*=2){
        int index = (tx+1)*2*up-1;
        if(index<BLOCK_DIM){
            partialSums[index] += partialSums[index-up];
        }
        __syncthreads();
    }

    //downsweep
    if(tx==0){
        partialSums[BLOCK_DIM-1]=0;
        __syncthreads();
    }
    for(int down=BLOCK_DIM/2; down>0; down/=2){
        int index = (tx+1)*2*down-1;

        //swap and add pattern
        if(index<BLOCK_DIM){
            float temp = partialSums[index-down];
            partialSums[index-down] = partialSums[index];
            partialSums[index] += temp;
        }
        __syncthreads();
    }
    if(tid<N){
        B[tid] = partialSums[tx];
    }
}