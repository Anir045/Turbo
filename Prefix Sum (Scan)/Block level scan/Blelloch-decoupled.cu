constexpr int BLOCK_DIM 256

enum TileStatus : int {
    INVALID = 0,
    AGGREGATE_READY = 1,
    PREFIX_READY = 2
};

struct TileState{
    float aggregate;
    float prefix;
    TileStatus status;
};

__device__ float local_scan(float *partialSums, int BLOCK_DIM, int tx){

    //upsweep
    for(int up=1; up<BLOCK_DIM; up<<=1){
        int index = (tx+1)*2*up-1;
        if(index<BLOCK_DIM){
            partialSums[index] += partialSums[index-up];
        }
        __syncthreads();
    }

    float local_aggregate = partialSums[BLOCK_DIM-1];
    partialSums[BLOCK_DIM-1] = 0;
    __syncthreads();

    //downsweep
    for(int down=BLOCK_DIM/2;down>0;down>>=1){
        int index = (tx+1)*2*down-1;
        if(index<BLOCK_DIM){
            float temp = partialSums[index-down];
            partialSums[index-down] = partialSums[down];
            partialSums[down] += temp;

        }
        __syncthreads();
    }
    return local_aggregate;
}

__global__ void scan_kernel(float *A, int N, TileState *state, float *B){
    int tile = blockIdx.x;
    int tx = threadIdx.x;
    __shared__ float partialSums[BLOCK_DIM];
    partialSums[tx] = A[tile*BLOCK_DIM+tx];

    __syncthreads();

    float local_aggregate = local_scan(partialSums, BLOCK_DIM, tx);

    //publish aggregate
    if(tx==0){
        state[tile].aggregate = local_aggregate;
        __threadfence();

        state[tile].status = AGGREGATE_READY;
    }
    __syncthreads();

    //Block 0  has no predecessor
    if(tile==0){
        if(tx==0){
            state[0].prefix=0;
            __threadfence();

            state[0].status = PREFIX_READY;
        }
        return;
    }


    //lookback
    if(tx==0){
        float prefix = 0.0f;
        int i=tile-1;
        
        while(true){
            TileState presentState = state[i].status;

            if(presentState==PREFIX_READY){
                prefix += state[i].prefix;
                break;
            }
            if(presentState==AGGREGATE_READY){
                prefix += state[i].aggregate;
                --i;
                continue;
            }
            if(presentState==INVALID){
                continue;
            }
        }

        //publish the learnt prefix
        state[tile].prefix = prefix;
        __threadfence();
        state[tile].status = PREFIX_READY;
    }
    __syncthreads();
    B[tile*BLOCK_DIM+tx] = partialSums[tx] + tile_prefix;

}