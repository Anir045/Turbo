//Corner Turning Transpose example
//Main idea, Coalesced Read from global to shared, compute in shared, Coalesced Write from shared to global
#include<cuda_runtime.h>
#include<stdio.h>

#define TS 32

__global__ void Transpose(const float *B, float *B_t, int M, int N){
  __shared__ float Bds[TS][TS+1];

  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  int y_in = by*TS+ty; int x_in = bx*TS+tx;

  if(y_in<M && x_in<N){
    Bds[ty][tx] = B[y_in*N +x_in];
  }
  __syncthreads();

  int y_out = bx*TS+ty; int x_out = by*TS+tx;
  if(y_out<N && x_out<M){
    B_t[y_out*M+x_out] = Bds[tx][ty];
  }
}

int main() {

    // Rectangular matrix: M rows, N columns
    const int M = 64;
    const int N = 96;
    size_t bytes = M * N * sizeof(float);

    float *h_B = (float*)malloc(bytes);
    float *h_Bt = (float*)malloc(bytes);

    float *d_B, *d_Bt;

    // Initialize input matrix
    for(int i= 0;i < M;i++) {
        for (int j=0; j<N; j++) {
            h_B[i * N + j] = (float)(i * N + j);
        }
    }

    // Allocate GPU memory
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_Bt, bytes);
    cudaMemcpy(d_B, h_B, bytes,cudaMemcpyHostToDevice);

    // Launch configuration
    dim3 block(TS, TS);
    dim3 grid((N + TS - 1)/TS, (M + TS - 1)/TS);

    Transpose<<<grid, block>>>(d_B, d_Bt, M, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_Bt, d_Bt, bytes,cudaMemcpyDeviceToHost);

    // Verify transpose
    bool correct = true;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {

            if (h_Bt[j * M + i] != h_B[i * N + j]) {
                correct = false;
                break;
            }
        }
    }

    printf("Input dimensions: %d x %d\n", M, N);
    printf("Output dimensions: %d x %d\n", N, M);
    printf("Transpose: %s\n", correct ? "PASSED" : "FAILED");

    // Print a small portion
    printf("\nInput (first 5x5):\n");

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++)
            printf("%8.0f ", h_B[i * N + j]);
        printf("\n");
    }

    printf("\nOutput (first 5x5):\n");

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++)
            printf("%8.0f ", h_Bt[i * M + j]);
        printf("\n");
    }

    cudaFree(d_B);
    cudaFree(d_Bt);

    free(h_B);
    free(h_Bt);

    return 0;
}