//A(MxK) B(NxK) B_t (KxN) C(NxN)
#include<stdio.h>
#include<cuda_runtime.h>
#define TS 2

__global__ void AxB_t(const int *A, const int *B, int *C, int M, int K, int N){
  __shared__ int Ads[TS][TS];
  __shared__ int Bds[TS][TS];
  int by = blockIdx.y; int bx = blockIdx.x;
  int ty = threadIdx.y; int tx = threadIdx.x;

  int c_row = by*TS+ty;
  int c_col = bx*TS+tx;
  int b_row = bx*TS+ty;
  int Pvalue = 0.0;

  for(int ph=0;ph<(K+TS-1)/TS;++ph){
    if(c_row<M && ph*TS+tx<K){
      Ads[ty][tx] = A[c_row*K + ph*TS+tx];
    }else{
      Ads[ty][tx] = 0.0f;
    }
    if(b_row< N && ph*TS+tx<K){
      Bds[tx][ty] = B[b_row*K + ph*TS+tx];
    }else{
      Bds[tx][ty] = 0.0f;
    }
    __syncthreads();
    for(int k=0;k<TS;k++){
      Pvalue += Ads[ty][k]*Bds[k][tx];
    }
    __syncthreads();
  }
  if(c_row<M && c_col<N){
    C[c_row*N+c_col] = Pvalue;
  }
  
}

int main() {
    const int M=3;
    const int N=4;
    const int K=5;
    int h_A[M * K];
    int h_B[N * K];
    int h_C[M * N];
    int h_ref[M * N];

    // Initialize A
    for (int i = 0; i < M * K; i++)
        h_A[i] = (int)(i + 1);

    // Initialize B
    for (int i = 0; i < N * K; i++)
        h_B[i] = (int)(i + 1);

    // CPU reference: C = A * B^T
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {

            int sum = 0.0f;

            for (int k = 0; k < K; k++) {
                sum += h_A[i * K + k] * h_B[j * K + k];
            }

            h_ref[i * N + j] = sum;
        }
    }

    int *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, M * K * sizeof(int));
    cudaMalloc(&d_B, N * K * sizeof(int));
    cudaMalloc(&d_C, M * N * sizeof(int));

    cudaMemcpy(d_A, h_A, M * K * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * K * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threads(TS, TS);
    dim3 blocks((N + TS - 1) / TS,
                (M + TS - 1) / TS);

    AxB_t<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("GPU output:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++)
            printf("%d  ", h_C[i * N + j]);
        printf("\n");
    }

    printf("\nCPU reference:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++)
            printf("%d  ", h_ref[i * N + j]);
        printf("\n");
    }

    bool correct = true;

    for (int i = 0; i < M * N; i++) {
        if (fabsf(h_C[i] - h_ref[i]) > 1e-5f) {
            correct = false;
            break;
        }
    }

    printf("\nResult: %s\n", correct ? "PASS" : "FAIL");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}