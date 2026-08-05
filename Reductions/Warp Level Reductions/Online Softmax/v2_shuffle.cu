//assumed TS=32
#define FULL_MASK 0xffffffff


struct runningPair{
    float running_max;
    float running_sum;
};

__global__ void softmaxOnline(float *A, float *B, int N){
    int bx = blockIdx.x; int tx = threadIdx.x;
    int tid = bx*blockDim.x+tx;

    for(int ph=0;ph<(N+TS-1)/TS;++ph){

        int idx = ph*TS+tx;
        float x1 = -CUDART_INT_F;

        if(idx<N){
            x1 = A[idx];
        }
        float tmax = x1;
        
        for(int off1 = 16; off1>0; off1>>=1){
            float x2 = __shfl_down_sync(FULL_MASK, x1, off1);
            tmax = fmaxf(x2, tmax);
        }

        tmax = __shfl_sync(FULL_MASK, tmax, 0);

        float e = expf(x1-tmax);

        float tsum = e;
        for(int off2 = 16; off2>0;off2>>=1){
            tsum += __shfl_down_sync(FULL_MASK, tsum, off2);
        }

    }
    if(tx == 0){

        float newMax = fmaxf(running.running_max,tileMax);

        float newSum =  running.running_sum*expf(running.running_max - newMax) + tileSum*expf(tileMax - newMax);

        running.running_max = newMax;
        running.running_sum = newSum;
    }
}