import numpy as np
import matplotlib.pyplot as plt
from jacobi import rotate

#Does the rotation matrix turn the off diagonal at the specified index to 0?
rng = np.random.default_rng(1)
A   = (rng.random((5,5))*5)-2.5
S   = (A+A.T)
print(rotate(S,0,1))
print()
