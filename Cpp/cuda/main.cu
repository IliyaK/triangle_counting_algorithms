#include "main.h"
#include "preprocessing.h"

#include "algorithms/paper1_algorithm/paper1_algorithm.h"

int main() {
    double sum = 0.0;
    std::vector<std::pair<int, int>> edgeList = edgeLine_parser("../graphs/facebook_combined.txt");

    for(int i = 0; i < 1; i++){
        sum += static_cast<double>(algorithm1_gpu("../graphs/facebook_combined.txt", edgeList));
//        sum += static_cast<double>(algorithm1_cpu("../graphs/facebook_combined.txt", edgeList));
    }
    sum = sum/2;
    std::cout << "Triangles found: " << sum << std::endl;
    return 0;

}
