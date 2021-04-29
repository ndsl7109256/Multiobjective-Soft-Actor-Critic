import numpy as np

for _ in range(16):
    a = np.random.rand(2)
    a = a.astype(float)
    a /= a.sum()

    print(a)
