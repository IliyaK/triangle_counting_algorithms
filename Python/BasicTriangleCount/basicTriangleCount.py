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


# this algorithm in heavily based on 1D SpGEMM (Sparse General Matrix-Matrix Multiplication)
def basic_triangle_count(matrix_a):
    """
    The Triangle-basic algorithm
    performs a distributed SpGEMM to compute L · U, where the
    ith processor sends all rows
    :param matrix_a: Adjacency Matrix (ndarray)
    :return:
    """
    count = 0
    n, m = matrix_a.shape
    # making diagonal masks to split matrix into upper and lower section
    matrix_L_mask = np.tril(np.ones((n,m), dtype=bool), k=-1)
    matrix_U_mask = np.tril(np.ones((n,m), dtype=bool), k=1)
    # splitting input matrix into upper and lower sections
    matrix_L = np.where(matrix_L_mask, matrix_a, 0)
    matrix_U = np.where(matrix_U_mask, matrix_a, 0)
    matrix_U = matrix_U.astype(np.int32)
    matrix_L = matrix_L.astype(np.int32)

    ######## pyCUDA #############
    mod = SourceModule("""
    __global__ void sum_rows(int *dest, int *a, int num_rows, int num_cols)
    {
      const int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < num_rows) {
        int row_sum = 0;
        for (int j = 0; j < num_cols; j++) {
          row_sum += a[i * num_cols + j];
        }
        dest[i] = row_sum;
      }
    }
    """)
    sum_rows = mod.get_function("sum_rows")
    num_rows, num_cols = matrix_U.shape

    dest = np.zeros((num_rows,), dtype=np.int32)

    block_size = 400
    grid_size = (num_rows + block_size - 1) // block_size

    # this accomplishes line 3 of the BasicTriangleCounting(A)
    sum_rows(
        drv.Out(dest), drv.In(matrix_U), np.int32(num_rows), np.int32(num_cols),
        block=(block_size, 1, 1), grid=(grid_size, 1))
    ######### pyCUDA end
    # this accomplishes line 4 and 5
    non_zero_mask = dest != 0
    non_zero_indices = np.where(non_zero_mask)

    """
    for each thread do:
        sum rum all of the rows in matrix_U
        Urowsum = SUM(Ui, rows)     # pulling rows from Ui and summing them, returning list
        J = NzIndices(Urowsum)  # seeing which rows have non-zero sums, returning their index
        # MPI_ means function requires communication
        JS = MPI_ALLGATHERV(J) # getting all of the J rows?
        for j in range(p):
            LSpack[j] = SpRef(Li, :, JS(j))
        LSrecv = MPI_ALLTOALL(LSpack)
        Lrecv = concat LSresv
        B = SpGEMM(Lrecv, Ui)
        Ci = matrix_a * B
        loclcnt = SUM(SUM(Ci, cols), rows)      # summing the entire matrix
    """
    return count


def mask():
    return


def spgemm():
    return


def nzincides():
    return


def mpi_allgatherv():
    return


def lspack():
    return

def spref():
    return

def concatenate():
    return


if __name__ == '__main__':
    res = preprocessing("../../graphs/facebook_combined.txt")
    if type(res) == np.ndarray:
        basic_triangle_count(res)
    else:
        print("Could not complete basic_triangle_count()", file=sys.stderr)
        exit(1)