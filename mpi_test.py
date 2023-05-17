import multiprocessing
import numpy as np

def fx(x):
    return x + 1

if __name__ == "__main__":
    results = np.zeros([5])
    pool = multiprocessing.Pool(5) # using 6 cores
    ind = range(5) # no. of trajectories
    results[ind] = pool.map(fx, ind) 
    print(results)