/*
This is an important experiment, showcases how different indexing has different effect on the reads
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define TS 2
#define N 4

__global__ void test_full_matrix(const int *B, int *Out, int *All_Bds, int mode) {
    __shared__ int Bds[TS][TS];

    int bx = blockIdx.x;  // 0 or 1
    int by = blockIdx.y;  // 0 or 1
    int tx = threadIdx.x; // 0 or 1
    int ty = threadIdx.y; // 0 or 1

    int row = 0, col = 0;

    // -------------------------------------------------------------
    // STEP 1: Choose how we calculate global (row, col) to read B
    // -------------------------------------------------------------
    if (mode == 1) {
        // MODE 1: Normal (by*TS+ty, bx*TS+tx) -> Coalesced
        row = by * TS + ty;
        col = bx * TS + tx;
    } 
    else if (mode == 2) {
        // MODE 2: Macro-Flip ONLY (bx*TS+ty, by*TS+tx) -> Coalesced!
        // Swaps which 2x2 block we grab, keeps threads horizontal
        row = bx * TS + ty;
        col = by * TS + tx;
    } 
    else if (mode == 3) {
        // MODE 3: Micro-Flip ONLY (by*TS+tx, bx*TS+ty) -> Uncoalesced
        // Keeps the 2x2 block in place, transposes threads inside it
        row = by * TS + tx;
        col = bx * TS + ty;
    } 
    else if (mode == 4) {
        // MODE 4: BOTH Flipped (bx*TS+tx, by*TS+ty) -> Uncoalesced
        // Swaps the 2x2 block AND transposes threads inside it
        row = bx * TS + tx;
        col = by * TS + ty;
    }
    else if (mode == 5) {
        // MODE 5: TRUE CORNER TURNING!
        // Read like Mode 2 (Coalesced!), then flip Bds[tx][ty] in Shared Memory!
        row = bx * TS + ty;
        col = by * TS + tx;
    }

    // Load from B into Shared Memory Bds[ty][tx]
    Bds[ty][tx] = B[row * N + col];
    __syncthreads();

    // Save a copy of what Bds[ty][tx] looks like right now
    int out_r = by * TS + ty;
    int out_c = bx * TS + tx;
    All_Bds[out_r * N + out_c] = Bds[ty][tx];

    // -------------------------------------------------------------
    // STEP 2: Write from Shared Memory to Out (at normal out_r, out_c)
    // -------------------------------------------------------------
    if (mode == 5) {
        // In Mode 5 (Corner Turning), we flip [tx][ty] when reading Bds!
        Out[out_r * N + out_c] = Bds[tx][ty];
    } else {
        // In Modes 1-4, we just dump Bds[ty][tx] straight out so you can see it
        Out[out_r * N + out_c] = Bds[ty][tx];
    }
}

void print_4x4(const char* title, const int *M) {
    printf("%s\n", title);
    for (int r = 0; r < N; ++r) {
        if (r == TS) printf("  ---------+---------\n");
        printf("  [ %2d  %2d | %2d  %2d ]", M[r*N+0], M[r*N+1], M[r*N+2], M[r*N+3]);
        if (r == 0) printf("   <-- Top Blocks:    (by=0,bx=0) | (by=0,bx=1)");
        if (r == 2) printf("   <-- Bottom Blocks: (by=1,bx=0) | (by=1,bx=1)");
        printf("\n");
    }
    printf("\n");
}

int main() {
    int h_B[N * N] = {
         1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16
    };
    int h_Out[N * N], h_All_Bds[N * N];

    int *d_B, *d_Out, *d_All_Bds;
    cudaMalloc(&d_B, N * N * sizeof(int));
    cudaMalloc(&d_Out, N * N * sizeof(int));
    cudaMalloc(&d_All_Bds, N * N * sizeof(int));

    cudaMemcpy(d_B, h_B, N * N * sizeof(int), cudaMemcpyHostToDevice);

    print_4x4("ORIGINAL MATRIX B (4x4):", h_B);

    dim3 grid(N / TS, N / TS); // 2x2 grid of blocks
    dim3 block(TS, TS);        // 2x2 threads per block

    const char* titles[5] = {
        "=== MODE 1: row = by*TS+ty, col = bx*TS+tx (Normal Copy, Coalesced) ===",
        "=== MODE 2: row = bx*TS+ty, col = by*TS+tx (Macro-Flip ONLY: Blocks Swap, Coalesced!) ===",
        "=== MODE 3: row = by*TS+tx, col = bx*TS+ty (Micro-Flip ONLY: Inside-Tile Transpose, Uncoalesced) ===",
        "=== MODE 4: row = bx*TS+tx, col = by*TS+ty (Full Naive Transpose: Both Swap, Uncoalesced) ===",
        "=== MODE 5: CORNER TURNING (Mode 2 Coalesced Read + Bds[tx][ty] Shared Mem Flip!) ==="
    };

    for (int mode = 1; mode <= 5; ++mode) {
        test_full_matrix<<<grid, block>>>(d_B, d_Out, d_All_Bds, mode);
        cudaDeviceSynchronize();

        cudaMemcpy(h_Out, d_Out, N * N * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_All_Bds, d_All_Bds, N * N * sizeof(int), cudaMemcpyDeviceToHost);

        printf("=========================================================================\n");
        printf("%s\n", titles[mode - 1]);
        if (mode == 5) {
            print_4x4("1) State of Shared Memory Bds[ty][tx] right after Coalesced Read (Notice it equals Mode 2!):", h_All_Bds);
            print_4x4("2) Final Output after flipping Bds[tx][ty] in Shared Memory (100% Transposed!):", h_Out);
        } else {
            print_4x4("Resulting 4x4 Matrix (All 4 blocks' Bds[ty][tx] stitched together):", h_Out);
        }
    }

    cudaFree(d_B); cudaFree(d_Out); cudaFree(d_All_Bds);
    return 0;
}