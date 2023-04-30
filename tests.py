from functions import *
import numpy as np
import matplotlib.pyplot as plt
def test_upper_triangular(A):
    V,R = get_V_and_R(A)
    print('Upper Triangular Test:',np.sum(np.tril(R,-1)**2))
    return R

def test_reconstruction(A):
    
    V,R = get_V_and_R(A)
    Q   = get_Q(V)
    return A - Q@R

def test_orthogonality(A):
    V,R = get_V_and_R(A)
    Q   = get_Q(V)
    return Q.T@Q

np.random.seed(123)
A = np.random.random((5,10))

fig,ax=plt.subplots()
R = ax.imshow(test_upper_triangular(A))
fig.colorbar(R)
ax.set(title=r'$R$')
plt.show()

fig, ax = plt.subplots()
QtQ = ax.imshow(test_orthogonality(A))
ax.set(title=r'$Q^TQ = I$')
fig.colorbar(QtQ)
plt.show()

fig, ax = plt.subplots()
reconstruction = ax.imshow(test_reconstruction(A))
ax.set(title=r'$A - QR$')
fig.colorbar(reconstruction)
plt.show()