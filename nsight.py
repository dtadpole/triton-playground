from numba import cuda, guvectorize, vectorize, void, int32, float64, uint32
import math
import numpy as np
np.random.seed(1)

@cuda.jit(lineinfo=True)
def axpy(r, a, x, y):
    i = cuda.grid(1)
    if i < len(r):
        r[i] = a * x[i] + y[i]

@vectorize([float64(float64, float64, float64)], target='cuda')
def axpy_vectorize(a, x, y):
    return a * x + y

def vectorize_add_vectors(N):
    x = np.random.random(N)
    y = np.random.random(N)
    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    d_r = cuda.device_array_like(d_x)
    a = 4.5

    d_r = axpy_vectorize(a, d_x, d_y)

    return d_r.copy_to_host()

result = vectorize_add_vectors(2 ** 20)

print(result)
