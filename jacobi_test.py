import numpy as np
import matplotlib.pyplot as plt
from jacobi import rotate

#Does the rotation matrix turn the off diagonal at the specified index to 0?
rng = np.random.default_rng(1)
A   = (rng.random((5,5))*5)-2.5
S   = (A+A.T)
i = 1
j = 2
rotated_S     = rotate(S,i=i,j=j)
evals,evecs   = np.linalg.eig(S)
print('Original S eigenvalues', evals)
evals,evecs   = np.linalg.eig(rotated_S)
print('New S eigenvalues', evals)
print(rotated_S[i,j])
print(rotated_S[j,i])

S

print()
