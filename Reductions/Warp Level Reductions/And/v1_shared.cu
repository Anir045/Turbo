__global__ void and(const int *A, int N, int *res){
    __shared__ int a[BLOCK_DIM];
}