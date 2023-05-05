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

def test_same_eigenvalues():
    '''
    Create a matrix A with a predetermined set of eigenvalues and eigenvectors
    Write an algorithm called, hessenberg(A) that will return an H that is almost upper triangular
    and has the same eigenvalues as A
    '''
    n = 200
    expected_eigenvalues = np.arange(n)+1
    np.random.seed(123)
    expected_eigenvectors= np.random.random((n,n))
    A = expected_eigenvectors @ np.diag(expected_eigenvalues) @ np.linalg.inv(expected_eigenvectors)
    V, H = hessenberg(A)
    observed_eigenvalues, observed_eigenvectors = np.linalg.eig(H)
    print(expected_eigenvalues, observed_eigenvalues)
    print('Are (almost) all the elements below the diagonal 0? ', np.sum(np.tril(H,-2)))
    return H

def test_hessenberg_reconstruction():
    n = 100
    expected_eigenvalues = np.arange(n)+1
    np.random.seed(123)
    expected_eigenvectors= np.random.random((n,n))
    A = expected_eigenvectors @ np.diag(expected_eigenvalues) @ np.linalg.inv(expected_eigenvectors)
    V, H = hessenberg(A)
    Q = get_Q(V)
    return A - Q@H@Q.T

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

fig, ax = plt.subplots()
hessenberg_matrix = ax.imshow(test_same_eigenvalues())
ax.set(title=r'$H$')
fig.colorbar(hessenberg_matrix)
plt.show()

A_minus_QHQt = test_hessenberg_reconstruction()
fig, ax = plt.subplots()
reconstruction = ax.imshow(test_hessenberg_reconstruction())
ax.set(title=r'$A-QHQ^T$')
fig.colorbar(reconstruction)
plt.show()