#ifndef SOLVER_H
#define SOLVER_H

#include <vector>

#if defined(_USE_PYBIND)

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#endif

namespace NNLS {

  void init(int nr, int nc);
  void finalize();
  void non_negative_least_squares(double * A, double * y, double * x, int num_rows, int num_cols, double epsilon = 1e-6);
  void non_negative_least_squares_reuse(double * A, double * y, double * x, int num_rows, int num_cols, double epsilon = 1e-6);
  
  bool initialized = false;

  int max_size_vector = 0;
  int max_size_matrix = 0;
  int max_lwork = 0;

  int num_R;
  int num_P;
  
  std::vector<int> R;
  std::vector<int> P;
  std::vector<int> ipiv;

  std::vector<double> w;
  std::vector<double> wr;
  std::vector<double> s;
  std::vector<double> sP;
  std::vector<double> APy;
  std::vector<double> work;
  
  std::vector<double> Ax;
  std::vector<double> AP;
  std::vector<double> APP;

  double timer[14];

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

  void solve_reuse(py::array_t<double> A_, py::array_t<double> b_, py::array_t<double> x_, double epsilon = 1e-6)
  {
    py::buffer_info info_A = A_.request();
    py::buffer_info info_b = b_.request();
    py::buffer_info info_x = x_.request();
    
    double * A = static_cast<double*>(info_A.ptr);
    double * b = static_cast<double*>(info_b.ptr);
    double * x = static_cast<double*>(info_x.ptr);
    
    const int num_rows = info_A.shape[0];
    const int num_cols = info_A.shape[1];
    
    non_negative_least_squares_reuse(A, b, x, num_rows, num_cols, epsilon);
  }
#endif
  
}

#if defined(_USE_PYBIND)

PYBIND11_MODULE(my_nnls_solver, m) {
  m.doc() = "Python interface to solver"; // Add a docstring to the module
  m.def("solve", &NNLS::solve, "Solve Ax=b for x",py::arg("A"),py::arg("b"),py::arg("x"),py::arg("epsilon") = 1e-6);
  m.def("solve_reuse", &NNLS::solve_reuse, "Solve Ax=b for x",py::arg("A"),py::arg("b"),py::arg("x"),py::arg("epsilon") = 1e-6);
  m.def("finalize", &NNLS::finalize, "Shutdown library");
}

#endif

#endif
