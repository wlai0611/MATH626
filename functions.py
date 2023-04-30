import numpy as np

norm = lambda x:np.sqrt(np.dot(x,x))

def get_V_and_R(A):
    '''
    Input: A matrix, A

        [ a1, a2, a3 ..., an ]
        a1 is the first column of A

    Output: 1) an Upper triangular matrix, R whose columns are reflections of the columns of A

               R = [Qa1, Qa2, Qa3 ..., Qan]
               Q is a reflector matrix, a1 is first column of original A

            2) V, another matrix containing the mirrors that reflected each (subdiags of) column of A 

    |
    |   |  
    |   |  |  
    x1  |  |  | 
    |   x2 |  | 
    |   |  x3 |
    |   |  |  x4

        xk         . mirror (it is sparkly because it was recently cleaned)
       /       .*  /
      /     .*    /
     /   .*      /
    / .*        /
    ------------> 
               reflection (||xk||e1)

    '''
    nrows, ncols = A.shape
    mirrors      = np.zeros(A.shape)
    reflections  = A.copy()
    for k in range(min(nrows,ncols)):
        subdiagonal  = reflections[k:,k]
        reflection   = np.zeros(nrows-k)
        reflection[0]= norm(subdiagonal)
        mirror       = subdiagonal + reflection
        mirror       = mirror/norm(mirror)
        mirrors[k:,k]= mirror
        reflections[k:,:] -= 2 * np.outer(mirror,mirror) @ reflections[k:,:]
    return mirrors, reflections

def get_Q(V):
    '''
    In the first step that we obtained R, we did
    Q3*Q2*Q1*A = R
    
    where Q1 was v1 * v1T

    Now, we want to get the Q in A=QR which is backwards: A = Q1*Q2*Q3*R
                                                              --------
                                                                  Q
    So in order to get that Q, we must do

    Q = (v1 * v1T)(v2 * v2T)(v3 * v3T)...(vn * vnT) * Identity Matrix
    So we must start with the last column of V
    1. (vn * vnT) * Identity Matrix
    2. (vn-1 *vn-1T) * (vn * vnT) * Identity Matrix
    3. (vn-2 *vn-2T) * (vn-1 *vn-1T) * (vn * vnT) * Identity Matrix
    ...
    n. (v1 *v1T ) * ... * (vn * vnT) * Identity Matrix
    '''
    nrows, ncols = V.shape
    Q = np.identity(nrows)
    for k in range(ncols-1,-1,-1): #start from column n-1, to column 0, Thank python for the indexing
        mirror = V[k:,k]
        Q[k:,:] -= 2*np.outer(mirror,mirror)@Q[k:,:]  #reflect every column across the mirror
    return Q

