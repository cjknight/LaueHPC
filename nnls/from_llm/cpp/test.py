
from scipy import optimize
import numpy as np
import my_nnls_solver as nnls

num_rows = 5
num_cols = 4

start = 1
stop = start + num_rows * num_cols
A = np.arange(start, stop).reshape((num_rows,num_cols))

start = stop
stop = start + num_rows
y = np.arange(start, stop)

print("A= ", A)
print("y= ", y)

print("scipy   :: x= ", optimize.nnls(A,y)[0])

x = np.zeros(num_cols)

nnls.solve(A, y, x)

print("nnls    :: x= ", x)

x = np.zeros(num_cols)

nnls.solve(A, y, x, epsilon = 1e-4)

print("nnls(e) :: x= ", x)
