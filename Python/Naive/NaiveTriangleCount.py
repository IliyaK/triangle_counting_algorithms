import sys
import time
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from misc import adjacency_matrix_converter


def begin(mode, location):
    vertices, adjacency_matrix = adjacency_matrix_converter(location)
    start = time.time()
    if mode == "CPU":
        res = naive_cpu(adjacency_matrix)
    else:
        res = naive_gpu(adjacency_matrix)

    print(f"Triangles counted: {res} (Runtime: {time.time()-start}) seconds")


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
