#include "../../preprocessing.h"  // getting common processes
#include <chrono>

// CUDA
__global__ void copyUpperLower(int* mat, int* upper, int* lower, int numVertices_edgeList) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < numVertices_edgeList && j < numVertices_edgeList) {
        if (i <= j) {
            upper[i * numVertices_edgeList + j] = mat[i * numVertices_edgeList + j];
            lower[j * numVertices_edgeList + i] = mat[i * numVertices_edgeList + j];
        }
    }
}
__global__ void matrixMultiplication(int* product, int* upper, int* lower, int numVertices_edgeList) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int sum = 0;
    for (int k = 0; k < numVertices_edgeList; ++k) {
        sum += upper[i * numVertices_edgeList + k] * lower[k * numVertices_edgeList + j];
    }
    product[i * numVertices_edgeList + j] = sum;
}
__global__ void matrixElementWiseMultiply(int* mat, int* product, int* result, int numVertices_edgeList) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < numVertices_edgeList && j < numVertices_edgeList) {
        result[i * numVertices_edgeList + j] = mat[i * numVertices_edgeList + j] * product[i * numVertices_edgeList + j];
    }
}
__global__ void sumResultMatrix(int* result, int* sum, int numVertices_edgeList) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < numVertices_edgeList && j < numVertices_edgeList) {
        atomicAdd(sum, result[i * numVertices_edgeList + j]);
    }
}
///////////////

int algorithm1_gpu(const std::string& filename, std::vector<std::pair<int, int>>& edges){
//    std::vector<std::pair<int, int>> edgeList = edgeLine_parser(filename);
    std::vector<std::pair<int, int>> edgeList = edges;
    int numVertices_edgeList = getNumberOfVertices(edgeList);

    int sum = 0;

    int size = numVertices_edgeList * numVertices_edgeList * sizeof(int);
    std::size_t arr_size = numVertices_edgeList * numVertices_edgeList;
    int *mat = new int[arr_size]();

    // making adjacency matrix
    for (const auto &edge : edgeList) {
        int vertex1 = edge.first;
        int vertex2 = edge.second;

        // Set the elements to 1 to indicate the presence of an edge
        mat[vertex1 * numVertices_edgeList + vertex2] = 1;
        mat[vertex2 * numVertices_edgeList + vertex1] = 1;
    }
    std::cout << "graph parsed" << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    int *upper = new int[arr_size]();
    int *lower = new int[arr_size]();

    // splitting into upper and lower

    // Allocate device memory for mat, upper, and lower
    int* d_mat;
    int* d_upper;
    int* d_lower;

    cudaMalloc((void**)&d_mat, size);
    cudaMalloc((void**)&d_upper, size);
    cudaMalloc((void**)&d_lower, size);

    // Copy data from host to device
    cudaMemcpy(d_mat, mat, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions for the CUDA kernel
    dim3 dimGrid4((numVertices_edgeList + 15) / 16, (numVertices_edgeList + 15) / 16);
    dim3 dimBlock4(16, 16);

    // Launch the CUDA kernel to copy upper and lower triangular elements
    copyUpperLower<<<dimGrid4, dimBlock4>>>(d_mat, d_upper, d_lower, numVertices_edgeList);

    // Copy the results back from device to host if needed
    cudaMemcpy(upper, d_upper, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(lower, d_lower, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_mat);
    cudaFree(d_upper);
    cudaFree(d_lower);


    int *product = new int[arr_size]();


    // multiply upper and lower
    int *d_product;

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_product, size);
    cudaMalloc((void**)&d_upper, size);
    cudaMalloc((void**)&d_lower, size);

    // Copy data from host to device
    cudaMemcpy(d_upper, upper, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lower, lower, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 dimGrid(numVertices_edgeList, numVertices_edgeList);
    dim3 dimBlock(1, 1); // Adjust the block size as needed

    // Launch the CUDA kernel
    matrixMultiplication<<<dimGrid, dimBlock>>>(d_product, d_upper, d_lower, numVertices_edgeList);

    // Copy the result back to the host
    cudaMemcpy(product, d_product, size, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_product);
    cudaFree(d_upper);
    cudaFree(d_lower);
    // multiplying upper and lower

    delete[] upper;
    delete[] lower;

    int *result = new int[arr_size]();

    int* d_result;

    cudaMalloc((void**)&d_mat, size);
    cudaMalloc((void**)&d_product, size);
    cudaMalloc((void**)&d_result, size);

    // Copy data from host to device
    cudaMemcpy(d_mat, mat, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_product, product, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions for the CUDA kernel
    dim3 dimGrid2((numVertices_edgeList + 15) / 16, (numVertices_edgeList + 15) / 16);
    dim3 dimBlock2(16, 16);

    // Launch the CUDA kernel
    matrixElementWiseMultiply<<<dimGrid2, dimBlock2>>>(d_mat, d_product, d_result, numVertices_edgeList);

    // Copy the result back from device to host
    cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_mat);
    cudaFree(d_product);
    cudaFree(d_result);

    // element wise multiplication of mat and product


    delete[] mat;
    delete[] product;


    // summing up the matrix


    // Allocate device memory for result and sum
    int* d_sum;

    cudaMalloc((void**)&d_result, size);
    cudaMalloc((void**)&d_sum, sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_result, result, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum, &sum, sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions for the CUDA kernel
    dim3 dimGrid3((numVertices_edgeList + 15) / 16, (numVertices_edgeList + 15) / 16);
    dim3 dimBlock3(16, 16);

    // Launch the CUDA kernel to compute the sum
    sumResultMatrix<<<dimGrid3, dimBlock3>>>(d_result, d_sum, numVertices_edgeList);

    // Copy the sum back from device to host
    cudaMemcpy(&sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_result);
    cudaFree(d_sum);

    delete[] result;
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    long long elapsed_time = duration.count();
    std::cout << "GPU time run (milliseconds): " << elapsed_time << std::endl;
    return sum;
}

int algorithm1_cpu(const std::string& filename, std::vector<std::pair<int, int>>& edges){
//    std::vector<std::pair<int, int>> edgeList = edgeLine_parser(filename);
    std::vector<std::pair<int, int>> edgeList = edges;
    int numVertices_edgeList = getNumberOfVertices(edgeList);

    int sum = 0;

    int size = numVertices_edgeList * numVertices_edgeList * sizeof(int);
    std::size_t arr_size = numVertices_edgeList * numVertices_edgeList;
    int *mat = new int[arr_size]();
    int *upper = new int[arr_size]();
    int *lower = new int[arr_size]();
    int *product = new int[arr_size]();
    int *result = new int[arr_size]();
    // making adjacency matrix
    for (const auto &edge : edgeList) {
        int vertex1 = edge.first;
        int vertex2 = edge.second;

        // Set the elements to 1 to indicate the presence of an edge
        mat[vertex1 * numVertices_edgeList + vertex2] = 1;
        mat[vertex2 * numVertices_edgeList + vertex1] = 1;
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numVertices_edgeList; i++) {
        for (int j = 0; j < numVertices_edgeList; j++) {
            if (i <= j) {
                upper[i * numVertices_edgeList + j] = mat[i * numVertices_edgeList + j];
                lower[j * numVertices_edgeList + i] = mat[i * numVertices_edgeList + j];
            }
        }
    }

    for (int i = 0; i < numVertices_edgeList; ++i) {
        for (int j = 0; j < numVertices_edgeList; ++j) {
            for (int k = 0; k < numVertices_edgeList; ++k) {
                product[i * numVertices_edgeList + j] += upper[i * numVertices_edgeList + k] * lower[k * numVertices_edgeList + j];
            }
        }
    }

    delete[] upper;
    delete[] lower;

    for (int i = 0; i < numVertices_edgeList; i++) {
        for (int j = 0; j < numVertices_edgeList; j++) {
            // Calculate the element-wise product and store it in the result matrix
            result[i * numVertices_edgeList + j] = mat[i * numVertices_edgeList + j] * product[i * numVertices_edgeList + j];
        }
    }

    for (int i = 0; i < numVertices_edgeList; i++) {
        for (int j = 0; j < numVertices_edgeList; j++) {
            sum += result[i * numVertices_edgeList + j];
        }
    }

    delete[] result;
    delete[] product;
    delete[] mat;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    long long elapsed_time = duration.count();
    std::cout << "GPU time run (milliseconds): " << elapsed_time << std::endl;
    return sum;
}