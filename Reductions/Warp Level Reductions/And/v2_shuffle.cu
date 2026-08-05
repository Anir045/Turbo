#define FULL_MASK 0xffffffff

__global__ void and(const int *A, int N, int *res){
    __shared__ unsigned int a[BLOCK_DIM];
    int tx = threadIdx.x; int bx = blockIdx.x;
    int tid = bx*blockDim.x+tx;
    int stride = blockDim.x*gridDim.x;
    unsigned int t = 0xffffffff;

    for(int i=tid;i<N;i+=2*stride){
        t &= A[i];
        if(i+stride<N){
            t &= A[i+stride];
        }
    }
    a[tx] = t;
    __syncthreads();

    for(int k=blockDim.x/2;k>=64;k>>=1){
        if(tx<k){
            a[tx] &= a[tx+k];
        }
        __syncthreads();
    }

    if(tx<32){
        unsigned int temp = a[tx]&a[tx+32];
        for(int off=16; off>0; off>>=1){
            unsigned int val = __shfl_down_sync(FULL_MASK, temp, off);
            temp &= val;
        }
        if(tx==0){
            res[bx] = temp;
        }
    }
}