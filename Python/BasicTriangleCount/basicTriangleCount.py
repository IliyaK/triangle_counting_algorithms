"""
Implementation of Basic Triangle Count algorithm from A. Azad, A. Buluc ̧, and J. Gilbert, “Parallel triangle counting and enumeration using matrix algebra,” in 2015
IEEE International Parallel and Distributed Processing Symposium Workshop, pp. 804–811, IEEE, 2015.
This is a tester linear implementation, full GPU implementation will be done in C++ and CUDA

@author: Iliya Kulbaka
@date: Sept. 23, 2023
"""
import sys

import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from misc import adjacency_matrix_converter


def preprocessing(path_to_file):
    """
    Follows preprocessing routine described in the paper Experimental Setup - Dataset
    to preprocess the undirected graph into the input for basic_triangle_count()
    :return: Adjacency Matrix (ndarray) if all good False (bool) if file not found
    """
    # getting adjacency matrix from the undirected graph
    res = adjacency_matrix_converter(path_to_file)
    # making sure file exists
    if res:
        num_vertices, adjacency_matrix = res
        # symmetrizing the adjacency matrix
        adjacency_matrix = np.maximum(adjacency_matrix, adjacency_matrix.T)
        # removing diagonal (self-loops)
        np.fill_diagonal(adjacency_matrix, 0)
        # random permutation of rows and columns
        permutation = np.random.permutation(len(num_vertices))
        adjacency_matrix = adjacency_matrix[permutation][:, permutation]
        return adjacency_matrix
    else:
        return False


def basic_triangle_count(matrix_a):
    """
    :param matrix_a: Adjacency Matrix (ndarray)
    :return:
    """
    count = 0
    n, m = matrix_a.shape
    # making diagonal masks to split matrix into upper and lower section
    matrix_L_mask = np.tril(np.ones((n,m), dtype=bool), k=-1)
    matrix_U_mask = ~np.tril(np.ones((n,m), dtype=bool), k=0)
    # splitting input matrix into upper and lower sections
    matrix_L = np.where(matrix_L_mask, matrix_a, 0).astype(np.int32)
    matrix_U = np.where(matrix_U_mask, matrix_a, 0).astype(np.int32)
    del matrix_L_mask, matrix_U_mask, n, m

    ######## pyCUDA #############
    mod = SourceModule("""
    __global__ void nzindices(int *dest, int *u, int num_rows, int num_cols)
    {
      const int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < num_rows) {
        int row_sum = 0;
        for (int j = 0; j < num_cols; j++) {
          row_sum += u[i * num_cols + j];
        }
        if (row_sum != 0){
            dest[i] = 1;
        }
      }
    }
    """)
    sum_rows = mod.get_function("nzindices")
    num_rows, num_cols = matrix_U.shape

    dest = np.zeros((num_rows,), dtype=np.int32)

    block_size = 400
    grid_size = (num_rows + block_size - 1) // block_size

    # this accomplishes line 3-5 of the BasicTriangleCounting(A)
    sum_rows(
        drv.Out(dest), drv.In(matrix_U), np.int32(num_rows), np.int32(num_cols),
        block=(block_size, 1, 1), grid=(grid_size, 1))
    ######### pyCUDA end
    j = np.where(dest == 1)[0]
    j = j.astype(np.int32)

    ######## pyCUDA
    mod = SourceModule("""
    __global__ void extractRows(float* input_matrix, int* row_indices, float* output_matrix, int num_rows, int num_cols) {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x < num_cols){
            for(int i = 0; i<num_rows;i++){
                output_matrix[x+num_cols*i] = input_matrix[row_indices[x]*num_rows + i];
            }
        }
    }
    """)
    num_cols = len(j)

    dest = np.zeros((matrix_U.shape[1], num_cols), dtype=np.int32)

    block_size = 10
    grid_size = (num_cols + block_size - 1) // block_size
    extractColumns = mod.get_function("extractRows")

    extractColumns(drv.In(matrix_U), drv.In(j), drv.Out(dest), np.int32(matrix_U.shape[1]), np.int32(num_cols),
                   block=(block_size, 1, 1), grid=(grid_size, 1))

    matrix_U = dest.astype(np.int32)
    ######## pyCUDA end

    ####### pyCUDA
    # pulling columns
    mod = SourceModule("""
    __global__ void extractColumns(int* input_matrix, int* row_indices, int* output_matrix, int num_rows) {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x < num_rows){
            for (int i = 0; i<num_rows; i++){
                    output_matrix[x*num_rows + i] = input_matrix[row_indices[x] + i*num_rows];
            }
        }
    }
    """)
    num_rows = matrix_L.shape[0]
    num_cols = len(j)

    dest = np.zeros((num_cols, matrix_L.shape[1]), dtype=np.int32)

    block_size = 10
    grid_size = (num_cols + block_size - 1) // block_size
    extractColumns = mod.get_function("extractColumns")

    extractColumns(drv.In(matrix_L), drv.In(j), drv.Out(dest), np.int32(num_rows),
                   block=(block_size, 1, 1), grid=(grid_size, 1))

    Lrecv = dest.astype(np.int32)
    ######## pyCUDA
    # dot product on vectors
    mod = SourceModule("""
    __global__ void matrix_multiply(int *L, int *U, int *output, int rows_L, int cols_L, int cols_U) {
        int row = threadIdx.y + blockIdx.y * blockDim.y;
        int col = threadIdx.x + blockIdx.x * blockDim.x;
    
        if (row < rows_L && col < cols_U) {
            int sum = 0;
            for (int k = 0; k < cols_L; k++) {
                sum += L[row * cols_L + k] * U[k * cols_U + col];
            }
            output[row * cols_U + col] = sum;
        }
    }
    """)
    rows_L, cols_L = matrix_U.shape
    rows_U, cols_U = Lrecv.shape
    output_matrix = np.zeros((rows_L, cols_U), dtype=np.int32)
    matrix_multiply_gpu = mod.get_function("matrix_multiply")
    block_dim = (16, 16, 1)
    grid_dim = (int(np.ceil(cols_U / block_dim[0])), int(np.ceil(rows_L / block_dim[1])), 1)
    matrix_multiply_gpu(drv.In(matrix_U), drv.In(Lrecv), drv.Out(output_matrix),
                        np.int32(rows_L), np.int32(cols_L), np.int32(cols_U),
                        block=block_dim, grid=grid_dim)
    B = output_matrix
    ######## pyCUDA end
    ######## pyCUDA
    mod = SourceModule("""
    __global__ void elementwise_multiply(int *A, int *B, int n) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int idy = threadIdx.y + blockIdx.y * blockDim.y;
        int index = idx + idy * n;
            
        if (idx < n && idy < n) {
            A[index] *= B[index];
        }
    }
    """)
    n = matrix_a.shape[0]
    matrix_a = matrix_a.astype(np.int32)
    B = B.astype(np.int32)
    block_dim = (16, 16, 1)
    grid_dim = (n // block_dim[0] + 1, n // block_dim[1] + 1, 1)
    matrix_a_gpu = drv.mem_alloc(matrix_a.nbytes)
    matrix_B_gpu = drv.mem_alloc(B.nbytes)

    # Transfer the matrices from CPU to GPU
    drv.memcpy_htod(matrix_a_gpu, matrix_a)
    drv.memcpy_htod(matrix_B_gpu, B)
    matrix_multiply_gpu = mod.get_function("elementwise_multiply")

    matrix_multiply_gpu(matrix_a_gpu, matrix_B_gpu, np.int32(n), block=block_dim, grid=grid_dim)
    drv.memcpy_dtoh(matrix_a, matrix_a_gpu)
    ######## pyCUDA end
    matrix_a = matrix_a.astype(np.int32)
    ####### pyCuda
    mod = SourceModule("""
    __global__ void row_sum(int *matrix, int *row_sums, int n) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int idy = threadIdx.y + blockIdx.y * blockDim.y;
        int index = idx + idy * n;
        
        if (idx < n && idy < n) {
            int value = matrix[index];
            atomicAdd(&row_sums[idx], value);
        }
    }
    """)
    row_sum_kernel = mod.get_function("row_sum")
    matrix_gpu = drv.mem_alloc(matrix_a.nbytes)
    drv.memcpy_htod(matrix_gpu, matrix_a)
    row_sums_gpu = drv.mem_alloc(matrix_a.shape[0] * matrix_a.itemsize)
    row_sums = np.zeros(matrix_a.shape[0], dtype=np.int32)
    drv.memcpy_htod(row_sums_gpu, row_sums)
    block_dim = (16, 16, 1)
    grid_dim = (matrix_a.shape[0] // block_dim[0] + 1, matrix_a.shape[0] // block_dim[1] + 1, 1)
    row_sum_kernel(matrix_gpu, row_sums_gpu, np.int32(n), block=block_dim, grid=grid_dim)
    drv.memcpy_dtoh(row_sums, row_sums_gpu)
    count = sum(row_sums)
    ####### pyCuda end

    return count


def basic_triangle_count_cpu(matrix_a):
    n, m = matrix_a.shape
    # making diagonal masks to split matrix into upper and lower section
    matrix_L_mask = np.tril(np.ones((n,m), dtype=bool), k=-1)
    matrix_U_mask = ~np.tril(np.ones((n,m), dtype=bool), k=0)
    # splitting input matrix into upper and lower sections
    matrix_L = np.where(matrix_L_mask, matrix_a, 0).astype(np.int32)
    matrix_U = np.where(matrix_U_mask, matrix_a, 0).astype(np.int32)
    del matrix_L_mask, matrix_U_mask, n, m

    # Find indices of rows with non-zero sums
    nonzero_rows = np.nonzero(np.sum(matrix_U, axis=1))[0]

    # Create a new array of zeros with the same shape as matrix_U
    arr_zeros_rows = np.zeros(matrix_U.shape)
    arr_zeros_cols = np.zeros(matrix_U.shape)

    # Fill arr_zeros with non-zero sum rows
    arr_zeros_rows[nonzero_rows, :] = matrix_U[nonzero_rows, :]
    arr_zeros_cols[:, nonzero_rows] = matrix_L[:, nonzero_rows]

    B = arr_zeros_rows @ arr_zeros_cols
    res = np.multiply(matrix_a, B)

    count = np.sum(res)
    return count


def naive_cpu(adjacency_matrix):
    triangle_count = 0

    # Iterate through all possible triangles
    for i in range(adjacency_matrix.shape[0]):
        for j in range(i + 1, adjacency_matrix.shape[1]):
            for k in range(j + 1, adjacency_matrix.shape[1]):
                # Check if there is an edge between each pair of vertices in the triangle
                if adjacency_matrix[i, j] and adjacency_matrix[j, k] and adjacency_matrix[k, i]:
                    triangle_count += 1
    return triangle_count


def naive_gpu(adjacency_matrix):
    size = adjacency_matrix.shape[0]

    cuda_kernel = """
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
"""
    # Compile the CUDA kernel
    mod = SourceModule(cuda_kernel)
    count_triangles_kernel = mod.get_function("naive_gpu")

    # Define block and grid dimensions
    block_size = (16, 16, 1)
    grid_size = ((size + block_size[0] - 1) // block_size[0], (size + block_size[1] - 1) // block_size[1])
    res = np.zeros(1, dtype=np.int32)
    count_triangles_kernel(drv.In(adjacency_matrix), drv.Out(res), np.int32(size),
                           block=block_size, grid=grid_size)
    return res[0]


if __name__ == '__main__':
    res = preprocessing("../../graphs/facebook_combined.txt")
    import time
    times = []
    if type(res) == np.ndarray:
        # res = np.array([
        #     [0, 1, 1, 0, 0],
        #     [1, 0, 1, 1, 0],
        #     [1, 1, 0, 1, 1],
        #     [0, 1, 1, 0, 1],
        #     [0, 0, 1, 1, 0]
        # ], dtype=np.int32)
        print(naive_gpu(res))

    else:
        print("Could not complete basic_triangle_count()", file=sys.stderr)
        exit(1)
