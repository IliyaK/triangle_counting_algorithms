cmake_minimum_required(VERSION 3.21)
project(Cpp CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(Cpp cuda/main.cu cuda/main.h
        cuda/preprocessing.cu cuda/preprocessing.h
        cuda/algorithms/paper1_algorithm/paper1_algorithm.cu cuda/algorithms/paper1_algorithm/paper1_algorithm.h
        cuda/algorithms/paper2_algorithm/paper2_algorithm.cu cuda/algorithms/paper2_algorithm/paper2_algorithm.h cuda/algorithms/naive/naive.cu cuda/algorithms/naive/naive.h)

set_target_properties(Cpp PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
