struct cos{
    float dotP;
    float l2A;
    float l2B;
};

__global__ void cos_sim(float *A, float *B, int N, cos cosSim){
    __shared__ cos dot[BLOCK_DIM];
    int bx = blockIdx.x; int tx = threadIdx.x;
    int tid = bx * blockDim.x + tx;
    int strides = blockDim.x * gridDim.x;
    cos sum;
    sum.dotP = 0.0f;
    sum.l2A = 0.0f;
    sum.l2B = 0.0f;

    //try to do all three stuff at one go!
    for(int i=tid;i<N;i+=2*strides){
        
        if(i<N){
            float a = A[i]; float b = B[i];
            sum.dotP+= a*b;
            sum.l2A += a*a;
            sum.l2B += b*b;
            
        }
        if(i+strides<N){
            float a = A[i+strides]; float b = B[i+strides];
            sum.dotP+= a*b;
            sum.l2A += a*a;
            sum.l2B += b*b;
        }
    }
    dot[tx] = sum;
    __syncthreads();
    for(int j=blockDim.x/2;j>0;j/=2){
        if(tx<j){
            dot[tx].dotP += dot[tx+j].dotP;
            dot[tx].l2A += dot[tx+j].l2A;
            dot[tx].l2B += dot[tx+j].l2B;
        }
        __syncthreads();
    }
    if(tx==0){
        cosSim[bx] = dot[0];
    }

}