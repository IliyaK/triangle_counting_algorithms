This is Python implementation section containing both GPU and CPU implementations.

pyCuda install (linux) guide https://wiki.tiker.net/PyCuda/Installation/Linux/

For installation of dependencies please use pip install -r requirements.txt

Along with pip install pycuda. To make sure PyCuda is working you need to install nvidia-cuda-toolkit, nvidia-cuda-dev and python-pycuda using apt-get.


File structure.

BasicTriangleCount folder contains code for Matrix Multiplication implementation of both CPU and GPU. 

Naive folder contains code for Naive triangle count implementation for bot CPU and GPU. 

Graphs folder contains sample graph. 


To run the code type python python_activate.py --graph_file=Path to graph --mode=GPU or CPU --algorithm=Matrix_Multiplication or Naive
