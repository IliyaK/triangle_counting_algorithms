#include "main.h"
#include "preprocessing.h"

#include "algorithms/paper1_algorithm/paper1_algorithm.h"



int main() {
    // stores graph transitions in edge list representation
    std::vector<std::pair<int, int>> edgeList;
    edgeList = edgeLine_parser("../graphs/facebook_combined.txt");

    int numVertices_edgeList = getNumberOfVertices(edgeList);
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
