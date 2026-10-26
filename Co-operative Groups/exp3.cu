#include<cuda_runtime.h>
#include<cooperative_groups.h>

namespace cg = cooperative_groups;

struct{
    float max;
    float min;
}

__global__ void MinMax(float *A, int N, float *d_block_mins, float *d_block_maxs, float *d_global_min, float *d_global_max){
    cg::thread_block block = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();

    int tid = grid.thread_rank();
    int tx = block.thread_rank();
    int bx = block.group_index().x;

    __shared__ float smin[256];
    __shared__ float smax[256];

    s_min[tx] = (tid<N)? A[tid] : CUDART_INF_F;
    s_max[tx] = (tid<N)? A[tid] : -CUDART_INF_F;
    block.sync();

    for(int offset=block.size()/2; offset>0;offset/=2){
        if(tx<offset){
            s_min[tx] = fminf(s_min[tx], s_min(tx+offset));
            s_max[tx] = fmaxf(s_max[tx], s_max[tx+offset]);
        }
        block.sync();
    }
    if(tx==0){
        d_block_mins[bx] = s_min[0];
        d_block_maxs[bx] = s_max[0];
    }

    grid.sync();

    if(bx==0){
        float b_min = (tx<grid.num_blocks())? d_block_mins[tx] : CUDART_INF_F;
        float b_max = (tx<grid.num_blocks())? d_block_mins[tx] : -CUDART_INF_F;

        s_min[tx] = b_min;
        s_max[tx] = b_max;

        for(int offset=block.size()/2; offset>0; offset/=2){
            
        }
    }
    


}