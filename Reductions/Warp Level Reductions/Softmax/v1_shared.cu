struct parts{
    float num;
    float deno;
}
__global__ void softmax(float *A, int N, float *max){
    __shared__ parts sm[BLOCK_DIM];
    int bx = blockIdx.x; int tx = threadIdx.x;
    int tid = bx*blockDim.x+tx;
    int strides = blockDim.x*gridDim.x;
    parts local;
    float *m = max;

    for(int i=tid;i<N;i+=2*strides){
        if(i<N){
            float x = A[i];
            x = x-m;
            local.deno += __expf(x);
            A[i] = __expf(x);

        }
        if(i+strides<N){
            float x = A[i+strides];
            x = x-m;
            local.deno += __expf(x);
            A[i+strides] = __expf(x);
        }
    }
    sm[tx] = local;
    __syncthreads();
    for(int j=blockDim.x/2;j>0;j/=2){
        if(tx<j){
            sm[tx].deno += sm[tx+j].deno;
        }
        __syncthreads();
    }
}