#include "main.h"
#include "preprocessing.h"

#include "algorithms/paper1_algorithm/paper1_algorithm.h"
#include "algorithms/naive/naive.h"
int main() {
    double sum = 0.0;
    std::vector<std::pair<int, int>> edgeList = edgeLine_parser("../graphs/facebook_combined.txt");


    // for naive
    int numVertices_edgeList = getNumberOfVertices(edgeList);

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

    auto start_time = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 1; i++){
        // Matrix Multiplication GPU
//        sum = static_cast<double>(algorithm1_gpu("../graphs/facebook_combined.txt", edgeList));
        // Matrix Multiplication CPU
//        sum = static_cast<double>(algorithm1_cpu("../graphs/facebook_combined.txt", edgeList));
        // Naive CPU
//        sum = static_cast<double>(naive_cpu(mat, numVertices_edgeList));
        // Naive GPU
        sum = static_cast<double>(naive_gpu(mat, numVertices_edgeList));
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    long long elapsed_time = duration.count();
    std::cout << "Runtime (milliseconds): " << elapsed_time << std::endl;
    std::cout << "Triangles found: " << sum << std::endl;
    return 0;

}
