#include "main.h"
#include "preprocessing.h"

int main() {
    // stores graph transitions in edge list representation
    std::vector<std::pair<int, int>> edgeList;
    edgeList = edgeLine_parser("../graphs/facebook_combined.txt");

    int numVertices = getNumberOfVertices(edgeList);
    std::vector<int> colPtr;
    std::vector<int> rowIndices;

    edgeListToCSC(edgeList, numVertices, colPtr, rowIndices);



    std::cout << "HI";
    return 0;
}
