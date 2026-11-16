//A(MxK) B(KxN) = C(MxN)
#include<stdio.h>
#include<cuda_runtime.h>

#define TS 2

__global__ void AxB_tiled_diff_dim(const int *A, const int *B, int *C, int M, int K, int N){
  __shared__ int Ads[TS][TS];
  __shared__ int Bds[TS][TS+1];

  int by = blockIdx.y; int bx = blockIdx.x;
  int ty = threadIdx.y; int tx = threadIdx.x;

  int row = by*TS+ty;
  int col = bx*TS+tx;
  int Pvalue = 0.0f;
  for(int ph=0;ph<((K+TS-1)/TS);++ph){
    if(row<M && ph*TS+tx<K){
      Ads[ty][tx] = A[row*K + ph*TS + tx];
    }else{
      Ads[ty][tx] = 0.0f;
    }
    if((ph*TS+ty)<K && col<N){
      Bds[ty][tx] = B[(ph*TS+ty)*N + col];
    }else{
      Bds[ty][tx] = 0.0f;
    }
    __syncthreads();

    for(int k=0;k<TS;k++){
      Pvalue += Ads[ty][k] * Bds[k][tx];
    }
    __syncthreads();
  }
  if(row<M && col<N){
    C[row*N+col] = Pvalue;
  }
}

void printMatrix(const char *title, int *M, int r, int c){
  printf("\n%s (%d x %d):\n", title, r, c);

  for(int i=0;i<r;i++){
    for(int j=0;j<c;j++){
      printf("%d  ", M[i*c+j]);
    }
    printf("\n");
  }
}

int main(){

    // Different, non-divisible dimensions
    const int M = 3;
    const int K = 5;
    const int N = 4;

    int h_A[M*K] = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15
    };

    int h_B[K*N] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
        17, 18, 19, 20
    };

    int h_C[M*N] = {0};
    int h_ref[M*N] = {0};

    int *d_A, *d_B, *d_C;

    size_t bytes_A = M*K*sizeof(int);
    size_t bytes_B = K*N*sizeof(int);
    size_t bytes_C = M*N*sizeof(int);

    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);

    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);

    // Launch configuration
    dim3 block(TS, TS);

    dim3 grid(
        (N+TS-1)/TS,
        (M+TS-1)/TS
    );

    AxB_tiled_diff_dim<<<grid, block>>>(
        d_A, d_B, d_C, M, K, N
    );

    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost);

    // CPU reference implementation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                h_ref[i*N+j] +=
                    h_A[i*K+k] * h_B[k*N+j];
            }
        }
    }

    printMatrix("A", h_A, M, K);
    printMatrix("B", h_B, K, N);
    printMatrix("GPU RESULT", h_C, M, N);
    printMatrix("CPU REFERENCE", h_ref, M, N);

    // Verify
    bool correct = true;

    for (int i = 0; i < M*N; i++) {
        if (h_C[i] != h_ref[i]) {
            correct = false;
            break;
        }
    }

    printf("\nVerification: %s\n",
           correct ? "PASSED" : "FAILED");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}