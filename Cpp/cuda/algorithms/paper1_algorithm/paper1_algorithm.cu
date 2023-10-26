#include "../../preprocessing.h"  // to get CSCMatrix struct

__global__ void countEdgesInRowKernel(const int* colPtr, int numVertices, int* result) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numVertices) {
        result[row] = colPtr[row + 1] - colPtr[row];
    }
}

void countEdgesInRowsCUDA(const CSCMatrix& CSC, int numVertices, int* result) {
    int* d_colPtr;
    int* d_result;

    // Allocate memory on the device
    cudaMalloc((void**)&d_colPtr, (CSC.colPtr.size()) * sizeof(int));
    cudaMalloc((void**)&d_result, numVertices * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_colPtr, &CSC.colPtr[0], (CSC.colPtr.size()) * sizeof(int), cudaMemcpyHostToDevice);

    // Define the block size and grid size
    int blockSize = 256;
    int gridSize = (numVertices + blockSize - 1) / blockSize;

    // Launch the CUDA kernel
    countEdgesInRowKernel<<<gridSize, blockSize>>>(d_colPtr, numVertices, d_result);

    // Copy the result back to the host
    cudaMemcpy(result, d_result, numVertices * sizeof(int), cudaMemcpyDeviceToHost);

    // Free allocated memory on the device
    cudaFree(d_colPtr);
    cudaFree(d_result);
}

__global__ void nzindices(int *dest, const int *u, int num_rows, int num_cols)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_rows) {
        if (u[i] != 0) {
            dest[i] = 1;
        }
        else{
            dest[i] = 0;
        }
    }
}
void NonZeroIndices(int numVertices, int* uRowSum, int* uNonZero) {
    int* d_uRowSum;
    int* d_uNonZero;

    // Allocate memory on the device
    cudaMalloc((void**)&d_uRowSum, numVertices * sizeof(int));
    cudaMalloc((void**)&d_uNonZero, numVertices * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_uRowSum, uRowSum, numVertices * sizeof(int), cudaMemcpyHostToDevice);

    // Define the block size and grid size for countEdgesInRowKernel
    int blockSize = 256;
    int gridSize = (numVertices + blockSize - 1) / blockSize;

    // Define the block size and grid size for nzindices
    gridSize = (numVertices + blockSize - 1) / blockSize;

    // Launch the nzindices kernel
    nzindices<<<gridSize, blockSize>>>(d_uNonZero, d_uRowSum, numVertices, numVertices);

    // Copy the result back to the host
    cudaMemcpy(uNonZero, d_uNonZero, numVertices * sizeof(int), cudaMemcpyDeviceToHost);

    // Free allocated memory on the device
    cudaFree(d_uRowSum);
    cudaFree(d_uNonZero);
}
