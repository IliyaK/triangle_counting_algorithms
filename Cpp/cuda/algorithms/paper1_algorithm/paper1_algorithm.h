#ifndef CPP_PAPER1_ALGORITHM_H
#define CPP_PAPER1_ALGORITHM_H

__global__ void copyUpperLower(int* mat, int* upper, int* lower, int numVertices_edgeList);
__global__ void matrixMultiplication(int* product, int* upper, int* lower, int numVertices_edgeList);
__global__ void matrixElementWiseMultiply(int* mat, int* product, int* result, int numVertices_edgeList);
__global__ void sumResultMatrix(int* result, int* sum, int numVertices_edgeList);

int algorithm1_gpu(const std::string& filename, std::vector<std::pair<int, int>>& edges);
int algorithm1_cpu(const std::string& filename, std::vector<std::pair<int, int>>& edges);
#endif //CPP_PAPER1_ALGORITHM_H
