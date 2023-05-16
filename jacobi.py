import numpy as np

def rotate(S,i,j):
    '''
    Modify ith and jth rows and ith and jth columns to zero out the i,j entry while preserving
    eigenvalues

    Parameters
    ----------
    S: numpy.array
      A symmetric matrix 
    i: int
      The row of the off-diagonal we wish to make a 0
    j: int
      The column of the off-diagonal we wish to make a 0
    Returns
    -------
    numpy.array
      The same symmetric matrix but after being left multiplied by a rotation matrix.  The
      rotation matrix will make the off diagonal at i,j 0.  Other off-diagonals will be effected.
    '''
    m, m = S.shape
    angle = np.arctan((2*S[i,j])/(S[j,j] - S[i,i]))/2 
    #rotator = np.identity(m)
    #rotator[i,i] = rotator[j,j] = np.cos(angle)
    #rotator[i,j] = np.sin(angle)
    #rotator[j,i] = -np.sin(angle)

    '''
    A = S.copy()
    A[i,:] = np.cos(angle)*S[i,:] - np.sin(angle)*S[j,:]
    A[j,:] = np.cos(angle)*S[j,:] + np.sin(angle)*S[i,:]

    B = A.copy()

    B[:,i] = np.cos(angle)*A[:,i] - np.sin(angle)*A[:,j]
    B[:,j] = np.cos(angle)*A[:,j] + np.sin(angle)*A[:,i]
    '''
    row_i = np.cos(angle)*S[i,:] - np.sin(angle)*S[j,:]
    row_j = np.cos(angle)*S[j,:] + np.sin(angle)*S[i,:] #we need to perform on original S unmodified by prior code
    S[i,:] = row_i
    S[j,:] = row_j

    col_i = np.cos(angle)*S[:,i] - np.sin(angle)*S[:,j]
    col_j = np.cos(angle)*S[:,j] + np.sin(angle)*S[:,i]#perform on the post-row-operation matrix
    S[:,i] = col_i
    S[:,j] = col_j
    return S