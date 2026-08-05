#define TS 32
#define FULL_MASK 0xffffffff

struct RunningPair{
    float running_max;
    float running_sum;
};

__global__ void onlineSoftmax(const float *A, float *B, int N){

    int tx = threadIdx.x;

    // Running state
    RunningPair running;
    running.running_max = -CUDART_INF_F;
    running.running_sum = 0.0f;

    // --------------------------------------------------
    // Process one tile at a time
    // --------------------------------------------------

    for(int ph=0; ph<(N+TS-1)/TS; ph++){

        //------------------------------------------
        // Stage 1 : Load this thread's value
        //------------------------------------------

        int idx = ph*TS + tx;

        float x = -CUDART_INF_F;

        if(idx < N)
            x = A[idx];

        //------------------------------------------
        // Stage 2 : Tile maximum
        //------------------------------------------

        float tileMax = x;

        for(int off=16; off>0; off>>=1){

            float other =
                __shfl_down_sync(FULL_MASK,tileMax,off);

            tileMax = fmaxf(tileMax,other);
        }

        // Broadcast lane0 result
        tileMax = __shfl_sync(FULL_MASK,tileMax,0);

        //------------------------------------------
        // Stage 3 : Tile exponential sum
        //------------------------------------------

        float e = expf(x - tileMax);

        float tileSum = e;

        for(int off=16; off>0; off>>=1){

            tileSum +=
                __shfl_down_sync(FULL_MASK,tileSum,off);
        }

        tileSum = __shfl_sync(FULL_MASK,tileSum,0);

        //------------------------------------------
        // Stage 4 : Merge running state
        //------------------------------------------

        if(tx==0){

            float newMax =
                fmaxf(running.running_max,
                      tileMax);

            float newSum =
                running.running_sum *
                expf(running.running_max-newMax)

                +

                tileSum *
                expf(tileMax-newMax);

            running.running_max = newMax;
            running.running_sum = newSum;
        }

        //------------------------------------------
        // Stage 5
        // Broadcast running state
        //------------------------------------------

        running.running_max =
            __shfl_sync(FULL_MASK,
                        running.running_max,
                        0);

        running.running_sum =
            __shfl_sync(FULL_MASK,
                        running.running_sum,
                        0);

    }

    //--------------------------------------------------
    // Final normalization
    //--------------------------------------------------

    int idx = tx;

    if(idx < N){

        B[idx] =
            expf(A[idx]-running.running_max)
            /
            running.running_sum;
    }

}