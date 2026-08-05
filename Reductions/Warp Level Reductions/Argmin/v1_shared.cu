struct minPair{
    float val;
    int index;
};

__global__ void argmin(float *A, int N, minPair *blockMin){
    __shared__ minPair amin[BLOCK_DIM];
    int tx = threadIdx.x ; int bx = blockIdx.x;
    int tid = bx*blockDim.x+tx;
    int strides = blockDim.x*gridDim.x;
    minPair min;
    min.val= FLT_MAX;;
    min.index = -1;

    for(int i=tid;i<N;i+=2*strides){
        if(i<N){
            if(min.val>A[i] || (min.val==A[i] && i<min.index)){
                min.val=A[i];
                min.index=i;
            }
        }
        if(i+strides<N){
            if(min.val>A[i+strides] || (min.val == A[i+strides] && i+strides<min.index)){
                min.val=A[i+strides];
                min.index=i+strides;
            }
        }
    }
    amin[tx] = min;
    __syncthreads();
    for(int j=blockDim.x/2;j>0;j/=2){
        if(tx<j){
            if(amin[tx].val>amin[tx+j].val){
                amin[tx].val = amin[tx+j].val;
                amin[tx].index = amin[tx+j].index;
            } 
            else if(amin[tx+j].val==amin[tx].val && amin[tx+j].index<amin[tx].index){
                amin[tx] = amin[tx+j];
            }
        }
        __syncthreads();
    }
    if(tx == 0){
        blockMin[bx] = amin[0];
    }

}