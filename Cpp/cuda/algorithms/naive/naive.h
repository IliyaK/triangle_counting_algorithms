#ifndef CPP_NAIVE_H
#define CPP_NAIVE_H
#include "../../preprocessing.h"  // getting common processes

int naive_cpu(int *adjacency_matrix, int numVertices);
int naive_gpu(int *adjacency_matrix, int numVertices);
#endif //CPP_NAIVE_H
