#include "preprocessing.h"

std::vector<std::pair<int, int>> edgeLine_parser(const std::string& file_name){
    std::vector<std::pair<int, int>> edgeList;

    std::ifstream inputFile(file_name);
    if(inputFile.is_open()){
        std::string line;
        // location of the space in the format int space int
        std::size_t found = 0;
        // digit 1
        std::string temp1;
        // digit 2
        std::string temp2;

        // parsing graph file into edge line format
        while(std::getline(inputFile, line)){
            found = line.find(' ');
            if (found != std::string::npos){
                for(int i=0; i<found; i++){
                    temp1 += line[i];
                }
                for(std::size_t i=found+1; i<line.length(); i++){
                    temp2 += line[i];
                }
                edgeList.emplace_back(std::stoi(temp1), std::stoi(temp2));
                temp1.clear();
                temp2.clear();
            }
        }
    } else {
        std::cerr << "Unable to open file.\n";
    }

    return edgeList;
}

int getNumberOfVertices(const std::vector<std::pair<int, int>>& edgeList) {
    int numVertices = 0;
    for (const auto& edge : edgeList) {
        numVertices = std::max(numVertices, std::max(edge.first, edge.second));
    }
    return numVertices + 1;
}

void edgeListToCSC(const std::vector<std::pair<int, int>>& edgeList, int numVertices, std::vector<int>& colPtr, std::vector<int>& rowIndices){
    std::size_t numEdges = edgeList.size();
    colPtr.resize(numVertices+1, 0);
    rowIndices.resize(numEdges, 0);

    // Step 1: Count the number of edges per column (vertex)
    for (int i = 0; i < numEdges; i++) {
        colPtr[edgeList[i].second + 1]++; // Increment the count for the column (destination vertex)
    }

    // Step 2: Convert counts in colPtr to column pointers
    for (int i = 1; i <= numVertices; i++) {
        colPtr[i] += colPtr[i - 1];
    }

    // Step 3: Populate rowIndices
    std::vector<int> colIndex(numVertices, 0); // Auxiliary array to track the index within each column
    for (int i = 0; i < numEdges; i++) {
        int destination = edgeList[i].second;
        rowIndices[colPtr[destination] + colIndex[destination]] = edgeList[i].first;
        colIndex[destination]++;
    }
}
