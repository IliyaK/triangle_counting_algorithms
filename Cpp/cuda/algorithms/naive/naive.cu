#include "naive.h"

int naive_cpu(int *adjacency_matrix, int numVertices) {
    int triangle_count = 0;

    // Iterate through all possible triangles
    for (int i = 0; i < numVertices; ++i) {
        for (int j = i + 1; j < numVertices; ++j) {
            for (int k = j + 1; k < numVertices; ++k) {
                // Check if there is an edge between each pair of vertices in the triangle
                if (adjacency_matrix[i * numVertices + j] &&
                    adjacency_matrix[j * numVertices + k] &&
                    adjacency_matrix[k * numVertices + i]) {
                    triangle_count += 1;
                }
            }
        }
    }
    std::cout << "Triangle count: " << triangle_count << std::endl;
    return triangle_count;
}


__global__ void naive_gpu(int *adjacency_matrix, int *triangle_count, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < size && j < size && i < j) {
        for (int k = j + 1; k < size; ++k) {
            if (adjacency_matrix[i * size + j] && adjacency_matrix[j * size + k] && adjacency_matrix[k * size + i]) {
                atomicAdd(triangle_count, 1);
            }
        }
    }
}

int naive_gpu(int *adjacency_matrix, int numVertices) {

    int*d_adjacency_matrix, *d_triangle_count;
    int triangle_count = 0;

    cudaMalloc((void**)&d_triangle_count, sizeof(int));
    cudaMemcpy(d_triangle_count, &triangle_count, sizeof(int), cudaMemcpyHostToDevice);

    // Allocate memory for adjacency_matrix on device and copy it
    cudaMalloc((void**)&d_adjacency_matrix, numVertices * numVertices * sizeof(int));
    cudaMemcpy(d_adjacency_matrix, adjacency_matrix, numVertices * numVertices * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((numVertices + blockSize.x - 1) / blockSize.x, (numVertices + blockSize.y - 1) / blockSize.y);
    naive_gpu<<<gridSize, blockSize>>>(d_adjacency_matrix, d_triangle_count, numVertices);

    // Copy result back to host
    cudaMemcpy(&triangle_count, d_triangle_count, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Triangle count: " << triangle_count << std::endl;

    return triangle_count;
}