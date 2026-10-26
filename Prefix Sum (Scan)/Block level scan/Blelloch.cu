struct TileState{
    float aggregate;
    float prefix;
    int state;
};
//State =0 : NOT_READY. State=1 : READY

__device__ float local_scan(float *partialSums, int BLOCK_DIM, int tx){
    //Blelloch scan here
    
    //upsweep
    for(int up=1;up<BLOCK_DIM;up<<=1){
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
            partialSums[index-down] = partialSums[index];
            partialSums[index] += temp;
        }
        __syncthreads();
    }
    return local_aggregate;
}

__global__ void scan_kernel(float *A, int N, TileState *state, float *B)
{
    int tile = blockIdx.x;
    int tileSize = BLOCK_DIM;
    __shared__ float partialSums[BLOCK_DIM];
    int tx = threadIdx.x;
    __shared__ float tile_prefix;
    partialSums[tx] = A[tile*BLOCK_DIM+tx];
    __syncthreads();

    

    float local_aggregate = local_scan(partialSums, BLOCK_DIM, tx);
    
    
    if(tx==0){
        
        if(tile==0){
        tile_prefix = 0.0f; 
        } else{
            while(atomicAdd(&state[tile-1].state, 0)==0);

            float previous_aggregate = state[tile-1].aggregate;
            tile_prefix = state[tile-1].prefix + previous_aggregate;
        }

        state[tile].aggregate = local_aggregate;
        state[tile].prefix = tile_prefix;

        __threadfence();

        atomicExch(&state[tile].state, 1);
   
    }
    __syncthreads();
    
    B[tile*BLOCK_DIM+tx] = partialSums[tx]+tile_prefix;
}