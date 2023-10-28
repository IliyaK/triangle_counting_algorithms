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
            if (line.empty()){
                continue;
            }
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
    return numVertices+1;
}

void edgeListToCSC(const std::vector<std::pair<int, int>>& edgeList,
                   int numVertices, CSCMatrix& CSC){
    std::size_t numEdges = edgeList.size();
    CSC.colPtr.resize(numVertices+1, 0);
    CSC.rowIndices.resize(numEdges, 0);

    // Step 1: Count the number of edges per column (vertex)
    for (int i = 0; i < numEdges; i++) {
        CSC.colPtr[edgeList[i].second + 1]++; // Increment the count for the column (destination vertex)
    }

    // Step 2: Convert counts in colPtr to column pointers
    for (int i = 1; i <= numVertices; i++) {
        CSC.colPtr[i] += CSC.colPtr[i - 1];
    }

    // Step 3: Populate rowIndices
    std::vector<int> colIndex(numVertices, 0); // Auxiliary array to track the index within each column
    for (int i = 0; i < numEdges; i++) {
        int destination = edgeList[i].second;
        CSC.rowIndices[CSC.colPtr[destination] + colIndex[destination]] = edgeList[i].first;
        colIndex[destination]++;
    }
}

void CSCToUpperAndLower(CSCMatrix& inputCSC,
                        CSCMatrix& upperCSC,
                        CSCMatrix& lowerCSC){

    const std::size_t numColumns = inputCSC.colPtr.size() -1;
    const std::size_t totalNonZeros = inputCSC.rowIndices.size();

    std::vector<int> tempUpperColPtr(numColumns + 1, 0);
    std::vector<int> tempLowerColPtr(numColumns + 1, 0);

    for (std::size_t i = 0; i < totalNonZeros; i++) {
        int column = inputCSC.rowIndices[i];
        if (column < numColumns / 2) {
            tempUpperColPtr[column + 1]++;
        } else {
            tempLowerColPtr[column - numColumns / 2 + 1]++;
        }
    }
    for (std::size_t i = 1; i <= numColumns; i++) {
        tempUpperColPtr[i] += tempUpperColPtr[i - 1];
        tempLowerColPtr[i] += tempLowerColPtr[i - 1];
    }

    std::vector<int> tempUpperRowIndices(tempUpperColPtr[numColumns]);
    std::vector<int> tempLowerRowIndices(tempLowerColPtr[numColumns]);

    for (std::size_t i = 0; i < totalNonZeros; i++) {
        int column = inputCSC.rowIndices[i];
        if (column < numColumns / 2) {
            tempUpperRowIndices[tempUpperColPtr[column]] = inputCSC.rowIndices[i];
            tempUpperColPtr[column]++;
        } else {
            tempLowerRowIndices[tempLowerColPtr[column - numColumns / 2]] = inputCSC.rowIndices[i] - numColumns / 2;
            tempLowerColPtr[column - numColumns / 2]++;
        }
    }

    upperCSC.colPtr = std::move(tempUpperColPtr);
    upperCSC.rowIndices = std::move(tempUpperRowIndices);

    lowerCSC.colPtr = std::move(tempLowerColPtr);
    lowerCSC.rowIndices = std::move(tempLowerRowIndices);

}

int getRowLength(const CSCMatrix& cscMatrix, int row) {
    if (row < 1 || row >= cscMatrix.colPtr.size()) {
        // Invalid row index
        return 0;
    }

    // Calculate the number of non-zero elements in the specified row
    int rowStart = cscMatrix.colPtr[row - 1];
    int rowEnd = cscMatrix.colPtr[row];
    int rowLength = rowEnd - rowStart;

    return rowLength;
}

int sumAlongRows(const CSCMatrix& cscMatrix, int startRow, int numRows) {
    int sum = 0;

    if (startRow < 1 || startRow > cscMatrix.colPtr.size() - 1) {
        // Invalid startRow
        return sum;
    }

    for (int row = startRow; row < startRow + numRows && row < cscMatrix.colPtr.size() - 1; row++) {
        int rowStart = cscMatrix.colPtr[row - 1];
        int rowEnd = cscMatrix.colPtr[row];

        for (int i = rowStart; i < rowEnd; i++) {
            // Add the non-zero value to the sum
            int value = cscMatrix.rowIndices[i];
            sum += value;
        }
    }

    return sum;
}