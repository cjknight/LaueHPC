
from scipy import optimize
import numpy as np
import cpp.my_nnls_solver as cpp_nnls
import eigen.solver as eigen_nnls

indx = 0
data_file = '../../../python/data/data-' + str(indx) + '.npy'
kernel_file = '../../../python/kernel/kernel-' + str(indx) + '.npy'

y = np.load(data_file)
A = np.load(kernel_file)

#A=np.array([[1,2],[3,4],[5,6]])
#y=np.array([5,11,17])

print("shape(A)= ", np.shape(A) )
print("shape(y)= ", np.shape(y) )

num_rows, num_cols = np.shape(A)

#print("A= ", A)
#print("y= ", y)

x0 = optimize.nnls(A, y)[0]

print("x(scipy)= ", x0)

Ae = np.array(A,dtype=np.float64,order='F')
ye = np.array(y,dtype=np.float64)
x2 = eigen_nnls.solve(Ae, ye)

print("x(eigen)= ", x2)

x1 = np.zeros(num_cols)
cpp_nnls.solve(A, y, x1)

print("x(cpp)= ", x1)


residual0 = np.linalg.norm(A.dot(x0) - y)
residual1 = np.linalg.norm(A.dot(x1) - y)
residual2 = np.linalg.norm(A.dot(x2) - y)

diff1 = sum( (x1 - x0) * (x1 - x0) )
diff2 = sum( (x2 - x0) * (x2 - x0) )
print("diff(cpp, eigen) = ", "{:e}".format(diff1), "{:e}".format(diff2))

print("residuals (scipy, cpp, eigen)= ", residual0, residual1, residual2)
