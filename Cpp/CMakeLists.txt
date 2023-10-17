cmake_minimum_required(VERSION 3.21)
project(Cpp CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(Cpp cuda/main.cu cuda/main.h cuda/preprocessing.cu cuda/preprocessing.h)

set_target_properties(Cpp PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
