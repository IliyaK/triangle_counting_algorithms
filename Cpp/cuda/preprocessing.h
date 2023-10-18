#include <fstream>
#include <string>
#include <vector>
#include <iostream>

#ifndef CPP_PREPROCESSING_H
#define CPP_PREPROCESSING_H

struct CSCMatrix {
    std::vector<int> colPtr;
    std::vector<int> rowIndices;
};

std::vector<std::pair<int, int>> edgeLine_parser(const std::string& file_name);

int getNumberOfVertices(const std::vector<std::pair<int, int>>& edgeList);

void edgeListToCSC(const std::vector<std::pair<int, int>>& edgeList,
                   int numVertices, CSCMatrix& CSC);

void CSCToUpperAndLower(CSCMatrix& inputCSC,
                        CSCMatrix& upperCSC,
                        CSCMatrix& lowerCSC);

int getRowLength(const CSCMatrix& cscMatrix, int row);

int sumAlongRows(const CSCMatrix& cscMatrix, int startRow, int numRows);

#endif //CPP_PREPROCESSING_H
