#ifndef CPP_PAPER1_ALGORITHM_H
#define CPP_PAPER1_ALGORITHM_H

void countEdgesInRowsCUDA(const CSCMatrix& CSC, int numVertices, int* result);
__global__ void countEdgesInRowKernel(const int* colPtr, int numVertices, int* result);

void NonZeroIndices(int numVertices, int* uRowSum, int* uNonZero);
__global__ void nzindices(int *dest, const int *u, int num_rows, int num_cols);

#endif //CPP_PAPER1_ALGORITHM_H
