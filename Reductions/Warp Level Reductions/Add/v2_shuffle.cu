#define FULL_MASK 0xffffffff

__global__ void warpadd(float *A, int N, float *res){
    __shared__ float a[BLOCK_DIM];
    int tx = threadIdx.x; int bx = blockIdx.x;
    int tid = bx*blockDim.x+tx;
    int strides = gridDim.x*blockDim.x;
    float sum =0.0f;
    
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

    for(int j=blockDim.x/2;j>=64;j>>=1){
        if(tx<j){
            a[tx] += a[tx+j];
        }
        __syncthreads();
    }
    if(tx < 32){

    float val = a[tx] + a[tx+32];

    val += __shfl_down_sync(0xffffffff,val,16);
    val += __shfl_down_sync(0xffffffff,val,8);
    val += __shfl_down_sync(0xffffffff,val,4);
    val += __shfl_down_sync(0xffffffff,val,2);
    val += __shfl_down_sync(0xffffffff,val,1);

    if(tx==0)
        res[bx]=val;
    }
}