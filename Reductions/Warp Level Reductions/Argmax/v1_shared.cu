struct maxPair{
    float val;
    int index;
};

__global__ void argmax(float *A, int N, maxPair *blockMax){
    __shared__ maxPair amax[BLOCK_DIM];
    int tx = threadIdx.x ; int bx = blockIdx.x;
    int tid = bx*blockDim.x+tx;
    int strides = blockDim.x*gridDim.x;
    maxPair max;
    max.val= -FLT_MAX;
    max.index = -1;

    for(int i=tid;i<N;i+=2*strides){
        if(i<N){
            if(max.val<A[i] || (max.val==A[i] && i<max.index)){
                max.val=A[i];
                max.index=i;
            }
        }
        if(i+strides<N){
            if(max.val<A[i+strides] || (max.val == A[i+strides] && i+strides<max.index)){
                max.val=A[i+strides];
                max.index=i+strides;
            }
        }
    }
    amax[tx] = max;
    __syncthreads();
    for(int j=blockDim.x/2;j>0;j/=2){
        if(tx<j){
            if(amax[tx].val<amax[tx+j].val){
                amax[tx].val = amax[tx+j].val;
                amax[tx].index = amax[tx+j].index;
            } 
            else if(amax[tx+j].val==amax[tx].val && amax[tx+j].index<amax[tx].index){
                amax[tx] = amax[tx+j];
            }
        }
        __syncthreads();
    }
    if(tx == 0){
        blockMax[bx] = amax[0];
    }

}