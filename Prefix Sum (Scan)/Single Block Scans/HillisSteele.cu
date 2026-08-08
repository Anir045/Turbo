__global__ void HilliSteele(float *A, int N, float *B){
    int bx = blockIdx.x; int tx = threadIdx.x;
    int tid = bx*blockDim.x+tx;
    float x = A[tid];
    int lane_id = tx%32;
    for(int off=1;off<=16;off<<=1){
        
        float y = __shfl_up_sync(0xffffffff, x, off);
        if (lane >= off){
            x += y;
        }
    }
    B[tid] = x;
}