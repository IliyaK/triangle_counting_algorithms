This is Python implementation section containing both GPU and CPU implementations.

To make sure CUDA is working you need to install nvidia-cuda-toolkit, nvidia-cuda-dev and python-pycuda using apt-get.
Along with having proper installation of CUDA, make sure nvcc folder is in the PATH variable, along with making sure that Cmake compiler has this variable set:\
-DCMAKE_CUDA_COMPILER="/path/to/cuda/bin/nvcc"

This project requires a minimum CMake version of 3.21