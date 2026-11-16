//NxN
#include<stdio.h>
#include<cuda_runtime.h>

#define TS 2

__global__ void AxB_tiled(const int *A, const int *B, int *C, int N){
  __shared__ int Ads[TS][TS];
  __shared__ int Bds[TS][TS+1];

  int by = blockIdx.y; int bx = blockIdx.x;
  int ty = threadIdx.y; int tx = threadIdx.x;

  int row = by*TS+ty;
  int col = bx*TS+tx;
  int Pvalue = 0.0f;
  for(int ph=0;ph<((N+TS-1)/TS);++ph){
    Ads[ty][tx] = A[row*N + ph*TS + tx];
    Bds[ty][tx] = B[(ph*TS+ty)*N + col];

    __syncthreads();

    for(int k=0;k<TS;k++){
      Pvalue += Ads[ty][k] * Bds[k][tx];
    }
    __syncthreads();
  }
  C[row*N+col] = Pvalue;
}

void printMatrix(const char *title, int *M, int N){
  printf("\n%s:\n", title);

  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      printf("%d  ", M[i*N+j]);
    }
    printf("\n");
  }
}

int main(){
  const int N=4;
  int h_A[N*N] = {
    1,2,3,4,
    5,6,7,8,
    9,10,11,12,
    13,14,15,16
  };
  int h_B[N*N] = {
    1,0,2,0,
    0,1,0,2,
    2,0,1,0,
    0,2,0,1
  };
  int h_C[N*N] = {0};
  int h_ref[N*N] = {0};

  int *d_A, *d_B, *d_C;
  size_t bytes = N*N*sizeof(int);

  cudaMalloc(&d_A, bytes);
  cudaMalloc(&d_B, bytes);
  cudaMalloc(&d_C, bytes);
  
  cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

  dim3 block(TS, TS);
  dim3 grid(N/TS, N/TS);

  AxB_tiled<<<grid, block>>>(d_A, d_B, d_C, N);

  cudaDeviceSynchronize();
  cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      for(int k=0;k<N;k++){
        h_ref[i*N+j] += h_A[i*N+k] *h_B[k*N+j];
      }
    }
  }
  printMatrix("A", h_A, N);
  printMatrix("B", h_B, N);
  printMatrix("GPU RESULT", h_C, N);
  printMatrix("CPU REFERENCE", h_ref, N);

  bool correct = true;

  for(int i=0;i<N*N;i++) {
      if (h_C[i]!=h_ref[i]){
        correct=false;
      }
  }

    printf("\nVerification: %s\n",correct ? "PASSED" : "FAILED");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;

}