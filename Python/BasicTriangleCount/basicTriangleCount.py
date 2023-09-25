"""
Implementation of Basic Triangle Count algorithm from A. Azad, A. Buluc ̧, and J. Gilbert, “Parallel triangle counting and enumeration using matrix algebra,” in 2015
IEEE International Parallel and Distributed Processing Symposium Workshop, pp. 804–811, IEEE, 2015.
This is a tester linear implementation, full GPU implementation will be done in C++ and CUDA

@author: Iliya Kulbaka
@date: Sept. 23, 2023
"""

import numpy as np
from misc import adjacency_matrix_converter
import sys

import pycuda

def preprocessing(path_to_file):
    """
    Follows preprocessing routine described in the paper Experimental Setup - Dataset
    to preprocess the undirected graph into the input for basic_triangle_count()
    :return: Adjacency Matrix (ndarray)
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
        permutation = np.random.permutation(num_vertices)
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
    :param matrix_a:
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
    # in this case 10 is the number of threads,
    # the loop spawns new thread and sections off the appropriate part of the matrix for calculation
    for i in range(10):
        row_sum_u = np.sum()
    """
    for each thread do:
        sum rum all of the rows in matrix_U
        Urowsum = SUM(Ui, rows)     # what I think is happening is that we are pulling rows from Ui and summing them
        j = NzIndices(Urowsum)
        JS = MPI_ALLGATHERV(J)
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
    res = preprocessing("./data.txt")
    if res:
        basic_triangle_count(res)
    else:
        print("Could not complete basic_triangle_count()", file=sys.stderr)
        exit(1)