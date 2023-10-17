#include <fstream>
#include <string>
#include <vector>
#include <iostream>

#ifndef CPP_PREPROCESSING_H
#define CPP_PREPROCESSING_H
std::vector<std::pair<int, int>> edgeLine_parser(const std::string& file_name);

int getNumberOfVertices(const std::vector<std::pair<int, int>>& edgeList);

void edgeListToCSC(const std::vector<std::pair<int, int>>& edgeList, int numVertices, std::vector<int>& colPtr, std::vector<int>& rowIndices);

#endif //CPP_PREPROCESSING_H
