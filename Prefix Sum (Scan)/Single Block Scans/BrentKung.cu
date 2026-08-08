//Assuming N is < BLOCK_DIM
#define BLOCK_DIM 256
__global__ void BrentKung(float *A, int N, float *B){
    int tx = threadIdx.x; int bx = blockIdx.x;
    int tid = bx*blockDim.x+tx;
    __shared__ float partialSums[BLOCK_DIM];

    //Load into shared
    if(tid<N){
        partialSums[tx] = A[tid];
    }else{
        partialSums[tx] = 0.0f;
    }
    __syncthreads();

    //upsweep reduction
    for(int up=1;up<BLOCK_DIM;up*=2){
        int index = (2*up)*(tx+1) -1;
        if(index<BLOCK_DIM){
            partialSums[index] += partialSums[index-up];
        }
        __syncthreads();
    }

    //downsweep 
    for(int down=BLOCK_DIM/4;down>0;down/=2){
        int index = (2*down)*(tx+1) -1 +down;
        if(index<BLOCK_DIM){
            partialSums[index] += partialSums[index-down];
        }
        __syncthreads();
    }
    if(tid<N){
        B[tid] = partialSums[tx];
    }
}