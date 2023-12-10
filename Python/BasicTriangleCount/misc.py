import os
import sys
# using this to make matrix operations easier
import numpy as np







def adjacency_matrix_converter(path_to_file):
    if os.path.isfile(path_to_file):
        # reading undirected graph file
        with open(path_to_file, "r") as fout:
            data = fout.read()

        # parsing the file to get all edges
        edges = data.split("\n")[:-1]
        graph = [tuple(e.split(" "))for e in edges]
        vertices = np.unique(graph)

        # Create an empty adjacency matrix filled with zeros
        adjacency_matrix = np.zeros((len(vertices), len(vertices)), dtype=np.ushort)

        # Populate the adjacency matrix based on the edges in the graph
        for u, v in graph:
            u_index = np.where(vertices == u)[0][0]
            v_index = np.where(vertices == v)[0][0]
            # For an undirected graph, set both (u,v) and (v,u) to 1
            adjacency_matrix[u_index][v_index] = 1
            adjacency_matrix[v_index][u_index] = 1

        return vertices, adjacency_matrix
    else:
        print(f"{path_to_file} ===== File Does Not Exist", file=sys.stderr)
        return False
