#include "main.h"
#include "preprocessing.h"

#include "algorithms/paper1_algorithm/paper1_algorithm.h"


struct adjacency_matrix{
    std::vector<std::vector<float>> matrix;
};




int main() {
    // stores graph transitions in edge list representation
//    std::vector<std::pair<int, int>> edgeList = {{1,2}, {1,7}, {7,2},
//                                                 {2,6}, {7,6}, {7,5},
//                                                 {2,3}, {5,3}, {5,4},
//                                                 {3,4}};
//    std::vector<std::pair<int, int>> edgeList = edgeLine_parser("../graphs/facebook_combined.txt");
    double sum = static_cast<double>(algorithm1_gpu("../graphs/facebook_combined.txt"));

    std::cout << sum/2 << std::endl;

    return 0;

}
