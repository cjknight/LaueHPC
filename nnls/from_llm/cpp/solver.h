
#if defined(_USE_PYBIND)

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#endif

namespace NNLS {

  void init(int nr, int nc);
  void non_negative_least_squares(double * A, double * y, double * x, int num_rows, int num_cols, double epsilon = 1e-6);
  
  bool initialized = false;

#if defined (_USE_PYBIND)
  void solve(py::array_t<double> A_, py::array_t<double> b_, py::array_t<double> x_, double epsilon = 1e-6)
  {
    py::buffer_info info_A = A_.request();
    py::buffer_info info_b = b_.request();
    py::buffer_info info_x = x_.request();
    
    double * A = static_cast<double*>(info_A.ptr);
    double * b = static_cast<double*>(info_b.ptr);
    double * x = static_cast<double*>(info_x.ptr);
    
    const int num_rows = info_A.shape[0];
    const int num_cols = info_A.shape[1];
    
    non_negative_least_squares(A, b, x, num_rows, num_cols, epsilon);
  }
#endif
  
}

#if defined(_USE_PYBIND)

PYBIND11_MODULE(my_nnls_solver, m) {
  m.doc() = "Python interface to solver"; // Add a docstring to the module
  m.def("solve", &NNLS::solve, "Solve Ax=b for x",py::arg("A"),py::arg("b"),py::arg("x"),py::arg("epsilon") = 1e-6);
}

#endif
