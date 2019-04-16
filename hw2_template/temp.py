import numpy as np
from numpy import linalg as la

np.set_printoptions(precision=4)

A = np.random.random_sample((5,5))
A[1:,1:] = 0.0

print(A)

e1 = np.zeros(5)
e1[0] = 1
e1 = e1.reshape((5,1))
c1 = A[:,0]

v = c1 - la.norm(c1)*e1
H = np.eye(5) - 2* (v@v.T) / (v.T @ v)
B =  A @ H
print(B)