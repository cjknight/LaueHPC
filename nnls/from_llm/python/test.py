
from scipy import optimize
import numpy as np
import cpp.my_nnls_solver as cpp_nnls
import eigen.solver as eigen_nnls

import time

def current_milli_time():
    return round(time.time() * 1000)

t_scipy = 0.0
t_cpp = 0.0
t_eigen = 0.0

ldiff_eigen = 0.0
ldiff_cpp = 0.0

for indx in range(100):
    print("indx= ", indx)
#    data_file = '../../../python/data/data-' + str(indx) + '.npy'
#    kernel_file = '../../../python/kernel/kernel-' + str(indx) + '.npy'
    
    data_file = '../../../../to-chris/data/data-' + str(indx) + '.npy'
    kernel_file = '../../../../to-chris/kernel/kernel-' + str(indx) + '.npy'

    y = np.load(data_file)
    A = np.load(kernel_file)
    
    num_rows, num_cols = np.shape(A)

    t0 = current_milli_time()
    x0 = optimize.nnls(A, y, atol=1e-4)[0] # succesfully completes first 115
    t_scipy += current_milli_time() - t0
    
    #print("x(scipy)= ", x0)

    Ae = np.array(A,dtype=np.float64,order='F')
    ye = np.array(y,dtype=np.float64)

    t0 = current_milli_time()
    x2 = eigen_nnls.solve(Ae, ye)
    t_eigen += current_milli_time() - t0

    #print("x(eigen)= ", x2)

    x1 = np.zeros(num_cols)

    t0 = current_milli_time()
    cpp_nnls.solve(A, y, x1)
    #cpp_nnls.solve(A, y, x1, epsilon = 1e-1)
    t_cpp += current_milli_time() - t0

    #print("x(cpp)= ", x1)

    residual0 = np.linalg.norm(A.dot(x0) - y)
    residual1 = np.linalg.norm(A.dot(x1) - y)
    residual2 = np.linalg.norm(A.dot(x2) - y)

    diff1 = sum( (x1 - x0) * (x1 - x0) )
    diff2 = sum( (x2 - x0) * (x2 - x0) )

    if diff1 > ldiff_cpp: ldiff_cpp = diff1
    if diff2 > ldiff_eigen: ldiff_eigen = diff2

    print("shape(A)= ", np.shape(A)," residuals (scipy, cpp, eigen)= ", residual0, residual1, residual2, "diff(cpp, eigen) = ", "{:e}".format(diff1), "{:e}".format(diff2))


print("t_scipy= ", t_scipy, " ms")
print("t_eigen= ", t_eigen, " ms")
print("t_cpp=   ", t_cpp, " ms")

print("Largest differences (cpp, eigen): ", "{:e}".format(ldiff_cpp), "{:e}".format(ldiff_eigen))

cpp_nnls.finalize()
