"""
code from https://documen.tician.de/pycuda/
"""

import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule


if __name__ == '__main__':

    mod = SourceModule("""
    __global__ void multiply_them(float *dest, float *a, float *b)
    {
      const int i = threadIdx.x;
      dest[i] = a[i] * b[i];
    }
    """)

    multiply_them = mod.get_function("multiply_them")

    a = np.random.randn(400).astype(np.float32)
    b = np.random.randn(400).astype(np.float32)

    dest = np.zeros_like(a)
    multiply_them(
        drv.Out(dest), drv.In(a), drv.In(b),
        block=(400,1,1), grid=(1,1))

    print(dest-a*b)
