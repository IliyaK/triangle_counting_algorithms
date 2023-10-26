#include "main.h"
#include "preprocessing.h"

#include "algorithms/paper1_algorithm/paper1_algorithm.h"


// TODO: ALL OF THESE FUNCTIONS ARE TESTS FOR ADJACENCY MATRIX
struct adjacency_matrix{
    std::vector<std::vector<float>> matrix;
};


// Function to convert an edge list to an adjacency matrix
adjacency_matrix edgeListToAdjacencyMatrix(const std::vector<std::pair<int, int>>& edgeList) {
    int n = 0;
    for (const auto& edge : edgeList) {
        n = std::max(n, std::max(edge.first, edge.second));
    }
    n++;

    adjacency_matrix adjMatrix;
    adjMatrix.matrix.resize(n, std::vector<float>(n, 0.0f));

    for (const auto& edge : edgeList) {
        adjMatrix.matrix[edge.first][edge.second] = 1;
        adjMatrix.matrix[edge.second][edge.first] = 1; // If the graph is undirected
    }

    return adjMatrix;
}

// Function to split an adjacency matrix into upper and lower parts
std::pair<adjacency_matrix, adjacency_matrix> splitAdjacencyMatrix(const adjacency_matrix& adjMatrix) {
    int n = adjMatrix.matrix.size();

    adjacency_matrix upperMatrix;
    upperMatrix.matrix.resize(n, std::vector<float>(n, 0.0f));

    adjacency_matrix lowerMatrix;
    lowerMatrix.matrix.resize(n, std::vector<float>(n, 0.0f));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i <= j) {
                upperMatrix.matrix[i][j] = adjMatrix.matrix[i][j];
            }
            if (i >= j) {
                lowerMatrix.matrix[i][j] = adjMatrix.matrix[i][j];
            }
        }
    }

    return std::make_pair(upperMatrix, lowerMatrix);
}

// Function to multiply the upper and lower parts of an adjacency matrix
adjacency_matrix multiplyUpperLower(const adjacency_matrix& upperMatrix, const adjacency_matrix& lowerMatrix) {
    int n = upperMatrix.matrix.size();

    adjacency_matrix productMatrix;
    productMatrix.matrix.resize(n, std::vector<float>(n, 0.0f));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                productMatrix.matrix[i][j] += upperMatrix.matrix[i][k] * lowerMatrix.matrix[k][j];
            }
        }
    }

    return productMatrix;
}


/////////////////////////////////////////////////
// CUDA parts
// CUDA kernel for matrix multiplication
//__global__ void matrixMultiplication(int n, float* upper, float* lower, float* result) {
//    int row = blockIdx.y * blockDim.y + threadIdx.y;
//    int col = blockIdx.x * blockDim.x + threadIdx.x;
//    float sum = 0.0f;
//
//    if (row < n && col < n) {
//        for (int k = 0; k < n; ++k) {
//            sum += upper[row * n + k] * lower[k * n + col];
//        }
//        result[row * n + col] = sum;
//    }
//}
//
//void multiplyUpperLower(const adjacency_matrix& upperMatrix, const adjacency_matrix& lowerMatrix, adjacency_matrix& productMatrix) {
//    int n = upperMatrix.matrix.size();
//
//    productMatrix.matrix.resize(n, std::vector<float>(n, 0.0f));
//
//    float* d_upperMatrix;
//    float* d_lowerMatrix;
//    float* d_productMatrix;
//
//    // Allocate memory on the GPU
//    cudaMalloc((void**)&d_upperMatrix, n * n * sizeof(float));
//    cudaMalloc((void**)&d_lowerMatrix, n * n * sizeof(float));
//    cudaMalloc((void**)&d_productMatrix, n * n * sizeof(float));
//
//    // Copy data from CPU to GPU
//    cudaMemcpy(d_upperMatrix, &upperMatrix.matrix[0][0], n * n * sizeof(float), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_lowerMatrix, &lowerMatrix.matrix[0][0], n * n * sizeof(float), cudaMemcpyHostToDevice);
//
//    // Calculate threadsPerBlock and numBlocks based on matrix dimensions
//    int maxThreadsPerBlock = 1024; // This is a common maximum for most GPUs
//    dim3 threadsPerBlock(maxThreadsPerBlock, 1);
//    dim3 numBlocks((n + maxThreadsPerBlock - 1) / maxThreadsPerBlock, n);
//
//    // Launch the CUDA kernel
//    matrixMultiplication<<<numBlocks, threadsPerBlock>>>(n, d_upperMatrix, d_lowerMatrix, d_productMatrix);
//
//    // Copy the result from GPU to CPU
//    cudaMemcpy(&productMatrix.matrix[0][0], d_productMatrix, n * n * sizeof(float), cudaMemcpyDeviceToHost);
//
//    // Free GPU memory
//    cudaFree(d_upperMatrix);
//    cudaFree(d_lowerMatrix);
//    cudaFree(d_productMatrix);
//}

/////////////////////////////////////////////////

// Function to perform element-wise matrix multiplication
adjacency_matrix elementWiseMatrixMultiplication(const adjacency_matrix& matrix1, const adjacency_matrix& matrix2) {
    int numRows = matrix1.matrix.size();
    int numCols = matrix1.matrix[0].size();

    adjacency_matrix resultMatrix;
    resultMatrix.matrix.resize(numRows, std::vector<float>(numCols, 0.0f));

    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            resultMatrix.matrix[i][j] = matrix1.matrix[i][j] * matrix2.matrix[i][j];
        }
    }

    return resultMatrix;
}

// Function to calculate the sum of elements in an adjacency matrix
int sumAdjacencyMatrix(const adjacency_matrix& adjMatrix) {
    int sum = 0;
    for (const auto& row : adjMatrix.matrix) {
        for (int val : row) {
            sum += val;
        }
    }
    return sum;
}



int main() {
    // stores graph transitions in edge list representation
    std::vector<std::pair<int, int>> edgeList = {{1,2}, {1,7}, {7,2},
                                                 {2,6}, {7,6}, {7,5},
                                                 {2,3}, {5,3}, {5,4},
                                                 {3,4}};
//    std::vector<std::pair<int, int>> edgeList = edgeLine_parser("../graphs/facebook_combined.txt");

    int numVertices_edgeList = getNumberOfVertices(edgeList);
//    std::cout<<numVertices_edgeList<<std::endl;

    // making adjacency matrix
    adjacency_matrix main_matrix = edgeListToAdjacencyMatrix(edgeList);

    std::pair<adjacency_matrix, adjacency_matrix> upperLowerMatrices = splitAdjacencyMatrix(main_matrix);
//
    adjacency_matrix upperMatrix = upperLowerMatrices.first;
    adjacency_matrix lowerMatrix = upperLowerMatrices.second;
//    adjacency_matrix productMatrix;
//
    adjacency_matrix productMatrix = multiplyUpperLower(upperMatrix, lowerMatrix);
//    multiplyUpperLower(upperMatrix, lowerMatrix, productMatrix);
    adjacency_matrix resultMatrix = elementWiseMatrixMultiplication(main_matrix, productMatrix);
    int final = sumAdjacencyMatrix(resultMatrix);
    std::cout<< final/2 <<std::endl;
    return 0;
    // TODO: TEST FOR ADJACENCY MATRIX ENDS IN LINE ABOVE
    CSCMatrix main_CSC;

    edgeListToCSC(edgeList, numVertices_edgeList, main_CSC);

    CSCMatrix upperCSC;
    CSCMatrix lowerCSC;

    CSCToUpperAndLower(main_CSC, upperCSC, lowerCSC);

    // summing rows of upper CSC
    const int number_of_vertices = upperCSC.colPtr.size() - 1;
    int* uRowSum = new int[number_of_vertices];
    countEdgesInRowsCUDA(upperCSC, number_of_vertices, uRowSum);

    int* uNonZero = new int[number_of_vertices];
    // NzIndices for uRowSum
    NonZeroIndices(number_of_vertices, uRowSum, uNonZero);

    return 0;
}
