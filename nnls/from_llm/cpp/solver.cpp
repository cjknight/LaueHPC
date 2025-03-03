#include <iostream>
#include <set>
#include <limits>

#include <omp.h>

#include "solver.h"

extern "C" {
  void dgels_(const char * trans, const int * m, const int * n, const int * nrhs,
	      double * A, const int * lda, double * B, const int * ldb, double * work,                     
	      int * lwork, int * info);

  void dsyrk_(const char * uplo, const char * trans, const int * n, const int * k, const double * alpha,
	      const double * a, const int * lda, const double * beta, double * c, const int * ldc);
  
  void dsytrf_(const char * uplo, const int * n, double * a, const int * lda,
	       int * ipiv, double * work, int * lwork, int * info);
  
  void dsysv_(const char * uplo, const int * n, const int * nrhs, double * a,
	      const int * lda, int * ipiv, double * b, const int * ldb,
	      double * work, int * lwork, int * info);

  void dgetrf_(int * m, int * n, double * A, int * lda, int * ipiv, int * info);

  void dgetrs_(char * trans, int * n, int * nrhs, double * A, int * lda, int * ipiv, 
	       double * B, int * ldb, int * info);
  
  void dgemv_(const char * trans, const int * m, const int * n, const double * alpha,
	      const double * a, const int * lda, const double * x, const int * incx,
             const double * beta, double * y, const int * incy);
  
  void dgemm_(const char * transa, const char * transb, const int * m, const int * n,
	      const int * k, const double * alpha, const double * a, const int * lda,
	      const double * b, const int * ldb, const double * beta, double * c,
	      const int * ldc);
}

using namespace NNLS;

void NNLS::init(int nr, int nc)
{
  double t0_ = omp_get_wtime();
  
  if(!initialized) {
    for(int i=0; i<14; ++i) timer[i] = 0.0;
  }
  
  int maxv = (nc > nr) ? nc : nr;
  if(maxv > max_size_vector) {
    max_size_vector = maxv + 100;

    R.resize(max_size_vector);
    P.resize(max_size_vector);

    ipiv.resize(max_size_vector);

    w.resize(max_size_vector);
    wr.resize(max_size_vector);
    s.resize(max_size_vector);
    sP.resize(max_size_vector);
    APy.resize(max_size_vector);
    Ax.resize(max_size_vector);

    x_old.resize(max_size_vector);
  }

  if(maxv*maxv > max_size_matrix) {
    max_size_matrix = maxv*maxv + 100;

    AP.resize(max_size_matrix);
    APP.resize(max_size_matrix);
  }

  int nrhs = 1;
  int lwork = -1;
  int info;
  
  double _work[1];
  
  dgels_((const char *) "N", &nr, &nr, &nrhs, nullptr, &nr, nullptr, &nr, &(_work[0]), &lwork, &info);
  
  lwork = static_cast<int>(_work[0] + 0.5);

  if(maxv * 64 > lwork) lwork = maxv * 64;
  
  if(lwork > max_lwork) max_lwork = lwork;
  work.resize(max_lwork);
  
  initialized = true;

  timer[1] += omp_get_wtime() - t0_;
}

void NNLS::finalize()
{
  R.clear();
  P.clear();
  ipiv.clear();

  w.clear();
  wr.clear();
  s.clear();
  sP.clear();
  APy.clear();

  Ax.clear();
  AP.clear();
  APP.clear();

  x_old.clear();
  
  printf("\nNNLS :: Timer Summary\n");
  printf(" -- i= %i  timer= %10.5f ms %s\n",0, timer[0]*1000.0, " :: NNLS non_negative_least_squares()");
  printf(" -- i= %i  timer= %10.5f ms %s\n",1, timer[1]*1000.0, " :: NNLS init()");
  printf(" -- i= %i  timer= %10.5f ms %s\n",2, timer[2]*1000.0, " :: NNLS before while-R loop");
  printf(" -- i= %i  timer= %10.5f ms %s\n",3, timer[3]*1000.0, " :: NNLS while-R loop");
  printf(" -- i= %i  timer= %10.5f ms %s\n",4, timer[4]*1000.0, " :: NNLS R :: before while-sP loop");
  printf(" -- i= %i  timer= %10.5f ms %s\n",5, timer[5]*1000.0, " :: NNLS R :: while-sP loop");
  printf(" -- i= %i  timer= %10.5f ms %s\n",6, timer[6]*1000.0, " :: NNLS R :: after while-sP loop");
  printf(" -- i= %i  timer= %10.5f ms %s\n",7, timer[7]*1000.0, " :: NNLS while-R loop :: setup");
  
  printf(" -- i= %i  timer= %10.5f ms %s\n",8, timer[8]*1000.0, " :: NNLS while-R loop :: takeP");
  printf(" -- i= %i  timer= %10.5f ms %s\n",9, timer[9]*1000.0, " :: NNLS while-R loop :: dgemm");
  printf(" -- i= %i  timer= %10.5f ms %s\n",10, timer[10]*1000.0, " :: NNLS while-R loop :: dgemv");
  printf(" -- i= %i  timer= %10.5f ms %s\n",11, timer[11]*1000.0, " :: NNLS while-R loop :: dgels");
  printf(" -- i= %i  timer= %10.5f ms %s\n",12, timer[12]*1000.0, " :: NNLS ");
  printf(" -- i= %i  timer= %10.5f ms %s\n",13, timer[13]*1000.0, " :: NNLS while-R loop :: takeP");
}

void NNLS::non_negative_least_squares(double * A, double * y, double * x, int num_rows, int num_cols, double epsilon)
{  
  //  if(!initialized)
  NNLS::init(num_rows, num_cols);
 
  double t0_ = omp_get_wtime();
  
  // int m = A.rows();
  // int n = A.cols();
  int m = num_rows;
  int n = num_cols;

  const double zero = 1e-15;
  
  int wwhile_count = 0;
  int swhile_count = 0;
  
  // VectorXd x = VectorXd::Zero(n);  // Initialize the solution vector x with zeros

  for(int i=0; i<num_cols; ++i) x[i] = 0.0;

  //    std::set<int> R;  // Initialize the set R with all indices
  //    for (int i = 0; i < n; ++i) {
  //        R.insert(i);
  //    }
  
  num_R = n;
  for(int i=0; i<num_R; ++i) R[i] = i;
  
  //    std::set<int> P;  // Initialize the set P to store selected indices
  
  num_P = 0;
  
  // VectorXd w = A.transpose() * (y - A * x);
  // x = 0, so x = A.transpose() * y
  
  // -- +++++++++++++++++++++++++++++++++++++

  // A.transpose * y
  
  {    
    const double alpha = 1.0;
    const double beta = 0.0;
    const int inc = 1;
    //printf("dgemv_(1) :: num_cols= %i  num_rows= %i  alpha= %f  inc= %i, beta= %f\n",num_cols, num_rows, alpha, inc, beta);
    dgemv_((const char *) "T", &num_cols, &num_rows, &alpha, A, &num_cols, y, &inc, &beta, w.data(), &inc);
  }
  
  // -- +++++++++++++++++++++++++++++++++++++
  
  // VectorXd wr(R.size());
  // int idx = 0;
  // for (auto ri = R.begin(); ri != R.end(); ri++) 
  // {
  //     wr[idx] = w[*ri];
  //     idx++;
  // }
  
  for(int i=0; i<num_R; ++i) wr[i] = w[ R[i] ];
  
  // // Create storage for AP
  // MatrixXd AP = A;

  // Create storage for s
  // VectorXd s = VectorXd::Zero(n);  // Initialize a vector s with zeros
  
  for(int i=0; i<n; ++i) s[i] = 0.0;
  
  int nrhs = 1;
  int info;
  
  // while (!R.empty() && wr.maxCoeff() > epsilon) {

  double wr_max = w[ R[0] ];
  for(int i=1; i<num_R; ++i) if(wr_max < w[ R[i] ]) wr_max = w[ R[i] ];

  double t1_ = omp_get_wtime();
  timer[2] += t1_ - t0_;
  
  while( (num_R > 0) && wr_max > epsilon) {
    double t2_ = omp_get_wtime();
    
    wwhile_count++;
    
    double max_dot_product = -std::numeric_limits<double>::infinity();

    int j_max = -1;

    //     for (int j : R) {
    //         double dot_product = w(j);
    //         if (dot_product > max_dot_product) {
    //             max_dot_product = dot_product;
    //             j_max = j;
    //         }
    //     }

    for(int j=0; j<num_R; ++j) {
      double dot_product = w[R[j]];
      //      printf(" -- j, w, max_dot_product, j_max= %i %f %f %i\n",R[j],w[R[j]],max_dot_product,j_max);
      if(dot_product > max_dot_product) {
	max_dot_product = dot_product;
	j_max = R[j];
      }
    }

    //     P.insert(j_max);  // Add the selected index to P

    P[num_P] = j_max;
    num_P++;
    
    //     R.erase(j_max);   // Remove the selected index from R
    
    int indx = 0;
    while(indx < num_R) {
      if(R[indx] == j_max) break;
      indx++;
    }
    for(int i=indx; i<num_R; ++i) R[i] = R[i+1];
    num_R--;

    double t7_ = omp_get_wtime();
    timer[7] += t7_ - t2_;
    
    //     //AP = AP.colwise().take(P);
    //     std::vector<int> pidx(P.begin(), P.end());
    //     //std::cout << " pidx size = " << pidx.size() << std::endl;
    //     // Construct the submatrix AP consisting of columns corresponding to indices in P
    //     AP = A(Eigen::placeholders::all, pidx);

    for(int i=0; i<num_P; ++i) {
      int indx = P[i];
      for(int j=0; j<num_rows; ++j) AP[j*num_P+i] = A[j*num_cols+indx];
    }

    double t8_ = omp_get_wtime();
    timer[8] += t8_ - t7_;
    
    //     VectorXd sP = (AP.transpose() * AP).ldlt().solve(AP.transpose() * y);  // Compute the least squares solution for the selected indices
    
    // APP = AP.transpose() * AP // (num_P x num_rows) * (num_rows x num_P) = num_P x num_P

    {
      const double alpha = 1.0;
      const double beta = 0.0;
      dgemm_((const char *) "N", (const char *) "T", &num_P, &num_P, &num_rows, &alpha, AP.data(), &num_P, AP.data(), &num_P, &beta, APP.data(), &num_P);
    }
    
#if 0
    if(num_P < 4) {
      //printf("num_rows= %i  num_P= %i\n",num_rows,num_P);
      // printf("AP= \n");
      // for(int i=0; i<num_rows; ++i) {
      // 	for(int j=0; j<num_P; ++j) printf(" %f",AP[i*num_P+j]);
      // 	printf("\n");
      // }
      
      printf("APP(dgemm)= \n");
      for(int i=0; i<num_P; ++i) {
	for(int j=0; j<num_P; ++j) printf(" %f",APP[i*num_P+j]);
	printf("\n");
      }
      for(int i=0; i<num_P*num_P; ++i) APP[i] = -1.0;

#if 1
      for(int i=0; i<num_P; ++i) {
	for(int j=0; j<=i; ++j) {
	  double val = 0.0;
	  for(int k=0; k<num_rows; ++k) val += AP[k*num_P+i] * AP[k*num_P+j];
	  APP[i*num_P+j] = val;
	  APP[j*num_P+i] = val;
	}
      }
#else
      {    
	const double alpha = 1.0;
	const double beta = 0.0;
	dsyrk_((const char *) "U", (const char *) "T", &num_P, &num_rows, &alpha, AP.data(), &num_rows, &beta, APP.data(), &num_P);
      }
#endif
      
      // fill upper APP
      
      printf("APP(dsyrk)= \n");
      for(int i=0; i<num_P; ++i) {
	for(int j=0; j<num_P; ++j) printf(" %f",APP[i*num_P+j]);
	printf("\n");
      }
      
      for(int i=0; i<num_P-1; ++i)
	for(int j=i+1; j<num_P; ++j) APP[i*num_P+j] = APP[j*num_P+i];
      
      printf("APP(fill)= \n");
      for(int i=0; i<num_P; ++i) {
	for(int j=0; j<num_P; ++j) printf(" %f",APP[i*num_P+j]);
	printf("\n");
      }
      
      //if(num_P == 4) exit(1);
    }
#endif
    
    double t9_ = omp_get_wtime();
    timer[9] += t9_ - t8_;
    
    // APy = AP.tranpose() * y // (num_P x num_rows) * num_rows

    for(int i=0; i<num_P; ++i) {
      double val = 0.0;
      for(int j=0; j<num_rows; ++j) val += AP[j*num_P+i] * y[j];
      APy[i] = val;
    }

    double t6_ = omp_get_wtime();
    timer[10] += t6_ - t9_;
#if 1
    // dsytrf_((const char *) "U", &num_P, APP, &num_P, ipiv, work, &max_lwork, &info);
    // dsysv_((const char *) "U", &num_P, &nrhs, APP, &num_P, ipiv, APy, &num_P, work, &max_lwork, &info);

    dgetrf_(&num_P, &num_P, APP.data(), &num_P, ipiv.data(), &info);
    dgetrs_((char *) "N", &num_P, &nrhs, APP.data(), &num_P, ipiv.data(), APy.data(), &num_P, &info);
#else
    dgels_((const char *) "N", &num_P, &num_P, &nrhs, APP.data(), &num_P, APy.data(), &num_P, work.data(), &max_lwork, &info);
#endif
    
    // printf("num_P= %i  APy= ",num_P);
    // for(int i=0; i<num_P; ++i) printf(" %f",APy[i]);
    // printf("\n");

    //    if(num_P == 2) exit(1);
    
    double t10_ = omp_get_wtime();
    timer[11] += t10_ - t6_;
    
    for(int i=0; i<num_P; ++i) sP[i] = APy[i];

    //     s = VectorXd::Zero(n);  // Initialize a vector s with zeros
    //     int idx = 0;
    //     for (auto pi = P.begin(); pi != P.end(); pi++) 
    //     {
    //         s[*pi] = sP(idx);
    //         idx++;
    //     }

    for(int i=0; i<num_P; ++i) s[ P[i] ] = sP[i];

    double min_sP = sP[0];
    for(int j=1; j<num_P; ++j) if(sP[j] < min_sP) min_sP = sP[j];
    
    double t3_ = omp_get_wtime();
    timer[4] += t3_ - t2_;
    
    while(min_sP < zero) {
      swhile_count++;
      
      double alpha = std::numeric_limits<double>::infinity();  // Initialize alpha as positive infinity

      //         for (int i : P) {
      //             if (s(i) <= 0) {
      //                 double alpha_candidate = x(i) / (x(i) - s(i));
      //                 if (alpha_candidate < alpha) {
      //                     alpha = alpha_candidate;
      //                 }
      //             }
      //         }
      
      for(int i=0; i<num_P; ++i) {
	if(s[P[i]] < zero) {
	  double alpha_candidate = x[P[i]] / (x[P[i]] - s[P[i]]);
	  if(alpha_candidate < alpha) alpha = alpha_candidate;
	}
      }

      //         // Update the solution vector x with the computed alpha
      //         x += alpha * (s - x);

      for(int i=0; i<num_cols; ++i) x[i] += alpha * (s[i] - x[i]);
      
      //         // Move indices from P to R if their corresponding elements in x become non-positive
      //         for (auto it = P.begin(); it != P.end();) {
      //             int i = *it;
      //             if (x(i) <= 0.0) {
      //                 it = P.erase(it);
      //                 R.insert(i);
      //             } else {
      //                 ++it;
      //             }
      //         }
      
      int ii = 0;
      while(ii < num_P) {
	if(x[P[ii]] < zero) {
	  int ii_ = P[ii];
	  
	  for(int i=ii; i<num_P; ++i) P[i] = P[i+1];
	  num_P--;

	  R[num_R] = ii_;
	  
	  num_R++;
	} else ii++;
      }
      
      //         // Recompute the submatrix AP and the least squares solution sP
      //         //AP = A;
      //         //AP = AP.colwise().take(P);
      //         std::vector<int> pidx(P.begin(), P.end());
      //         //AP = AP(pidx);
      //         AP = A(Eigen::placeholders::all, pidx); 

      for(int i=0; i<num_P; ++i) {
	int indx = P[i];
	for(int j=0; j<num_rows; ++j) AP[j*num_P+i] = A[j*num_cols+indx];
      }
      
      //         //s = VectorXd::Zero(n);
      //         s = VectorXd::Zero(n);  // Initialize a vector s with zeros
      
      for(int i=0; i<n; ++i) s[i] = 0.0;
      
      //         sP = (AP.transpose() * AP).ldlt().solve(AP.transpose() * y);
      
      // APP = AP.transpose() * AP // (num_P x num_rows) * (num_rows x num_P)

      {
	const double alpha = 1.0;
	const double beta = 0.0;
	dgemm_((const char *) "N", (const char *) "T", &num_P, &num_P, &num_rows, &alpha, AP.data(), &num_P, AP.data(), &num_P, &beta, APP.data(), &num_P);
      }
      
      // APy = AP.tranpose() * y

      {
	const double alpha = 1.0;
	const double beta = 0.0;
	const int inc = 1;
	//printf("dgemv_(2) :: num_P= %i  num_rows= %i  alpha= %f  inc= %i, beta= %f\n",num_P, num_rows, alpha, inc, beta);
	dgemv_((const char *) "N", &num_P, &num_rows, &alpha, AP.data(), &num_P, y, &inc, &beta, APy.data(), &inc);
      }
#if 1
    // dsytrf_((const char *) "U", &num_P, APP, &num_P, ipiv, work, &max_lwork, &info);
    // dsysv_((const char *) "U", &num_P, &nrhs, APP, &num_P, ipiv, APy, &num_P, work, &max_lwork, &info);

      dgetrf_(&num_P, &num_P, APP.data(), &num_P, ipiv.data(), &info);
      dgetrs_((char *) "N", &num_P, &nrhs, APP.data(), &num_P, ipiv.data(), APy.data(), &num_P, &info);
#else
      dgels_((const char *) "N", &num_P, &num_P, &nrhs, APP.data(), &num_P, APy.data(), &num_P, work.work(), &max_lwork, &info);
#endif
      
      for(int i=0; i<num_P; ++i) sP[i] = APy[i];
    
      //         int idx = 0;
      //         for (auto pi = P.begin(); pi != P.end(); pi++) 
      //         {
      //             s[*pi] = sP(idx);
      //             idx++;
      //         }
      
      for(int i=0; i<num_P; ++i) s[ P[i] ] = sP[i];
      
      min_sP = sP[0];
      for(int j=1; j<num_P; ++j) if(sP[j] < min_sP) min_sP = sP[j];
    } // while(min_sP)
    
    double t4_ = omp_get_wtime();
    timer[5] += t4_ - t3_;
    
    //     x = s;  // Update the solution vector x with the non-negative least squares solution

    for(int i=0; i<n; ++i) x[i] = s[i];
    
    //     w = A.transpose() * (y - A * x);

    // A * x
    {
      const double alpha = 1.0;
      const double beta = 0.0;
      const int inc = 1;
      //      printf("dgemv_(3) :: num_cols= %i  num_rows= %i  alpha= %f  inc= %i, beta= %f\n",num_cols, num_rows, alpha, inc, beta);
      dgemv_((const char *) "T", &num_cols, &num_rows, &alpha, A, &num_cols, x, &inc, &beta, Ax.data(), &inc);
    }
    
    // A.transpose * (y-Ax)
    {
      for(int i=0; i<m; ++i) Ax[i] = y[i] - Ax[i];
	
      const double alpha = 1.0;
      const double beta = 0.0;
      const int inc = 1;
      //      printf("dgemv_(4) :: n= %i  m= %i  alpha= %f  inc= %i, beta= %f\n",n, m, alpha, inc, beta);
      dgemv_((const char *) "N", &n, &m, &alpha, A, &n, Ax.data(), &inc, &beta, w.data(), &inc);
    }
    
    //     wr.resize(R.size());

    //     idx = 0;
    //     for (int i : R) {
    //         wr(idx) = w(i);
    //         ++idx;
    //     }

    for(int i=0; i<num_R; ++i) wr[i] = w[R[i]];
    
    wr_max = w[ R[0] ];
    for(int i=1; i<num_R; ++i) if(wr_max < w[ R[i] ]) wr_max = w[ R[i] ];
    
    double t5_ = omp_get_wtime();
    timer[6] += t5_ - t4_;
    
    if(wwhile_count > 200) {
      printf("NNLS :: WARNING!!! wwhile_count > %i and breaking loop\n",wwhile_count);
      break;
    }
  } // while(numR && max_wr)

  timer[3] += omp_get_wtime() - t1_;
  
  printf("NNLS::solve(cpp) -- num_R= %i  num_P= %i  w_count= %i  s_count= %i\n",num_R,num_P,wwhile_count,swhile_count);

  for(int i=0; i<num_cols; ++i) x_old[i] = x[i];
  
  timer[0] += omp_get_wtime() - t0_;
}

void NNLS::non_negative_least_squares_reuse(double * A, double * y, double * x, int num_rows, int num_cols, double epsilon)
{
  if(first_call) {
    non_negative_least_squares(A, y, x, num_rows, num_cols, epsilon);
    first_call = false;
    return;
  }
  
  //  if(!initialized)
  NNLS::init(num_rows, num_cols);

  //  printf("\nInside nnls_reuse()\n");
  
  double t0_ = omp_get_wtime();
  
  // int m = A.rows();
  // int n = A.cols();
  int m = num_rows;
  int n = num_cols;

  const double zero = 1e-15;
  
  int wwhile_count = 0;
  int swhile_count = 0;
  
  // VectorXd x = VectorXd::Zero(n);  // Initialize the solution vector x with zeros

#if 1
  for(int i=0; i<num_cols; ++i) x[i] = x_old[i];  
#else
  for(int i=0; i<num_cols; ++i) x[i] = 0.0;
#endif

  //    std::set<int> R;  // Initialize the set R with all indices
  //    for (int i = 0; i < n; ++i) {
  //        R.insert(i);
  //    }

  //  printf("(reuse) num_R= %i  num_P= %i\n",num_R,num_P);
  
#if 0
  num_R = n;
  for(int i=0; i<num_R; ++i) R[i] = i;

  //    std::set<int> P;  // Initialize the set P to store selected indices

  num_P = 0;
#endif

#if 1
    //     w = A.transpose() * (y - A * x);

    // A * x
    {
      const double alpha = 1.0;
      const double beta = 0.0;
      const int inc = 1;
      //      printf("dgemv_(3) :: num_cols= %i  num_rows= %i  alpha= %f  inc= %i, beta= %f\n",num_cols, num_rows, alpha, inc, beta);
      dgemv_((const char *) "T", &num_cols, &num_rows, &alpha, A, &num_cols, x, &inc, &beta, Ax.data(), &inc);
    }
    
    // A.transpose * (y-Ax)
    {
      for(int i=0; i<m; ++i) Ax[i] = y[i] - Ax[i];
	
      const double alpha = 1.0;
      const double beta = 0.0;
      const int inc = 1;
      //      printf("dgemv_(4) :: n= %i  m= %i  alpha= %f  inc= %i, beta= %f\n",n, m, alpha, inc, beta);
      dgemv_((const char *) "N", &n, &m, &alpha, A, &n, Ax.data(), &inc, &beta, w.data(), &inc);
    }
#else
  // VectorXd w = A.transpose() * (y - A * x);
  // x = 0, so w = A.transpose() * y
  
  // -- +++++++++++++++++++++++++++++++++++++

  // A.transpose * y
  
  {    
    const double alpha = 1.0;
    const double beta = 0.0;
    const int inc = 1;
    //    printf("dgemv_(r1) :: num_cols= %i  num_rows= %i  alpha= %f  inc= %i, beta= %f\n",num_cols, num_rows, alpha, inc, beta);
    dgemv_((const char *) "T", &num_cols, &num_rows, &alpha, A, &num_cols, y, &inc, &beta, w.data(), &inc);
  }
#endif
  
  // -- +++++++++++++++++++++++++++++++++++++
  
  // VectorXd wr(R.size());
  // int idx = 0;
  // for (auto ri = R.begin(); ri != R.end(); ri++) 
  // {
  //     wr[idx] = w[*ri];
  //     idx++;
  // }
  
  for(int i=0; i<num_R; ++i) wr[i] = w[ R[i] ];
  
  // // Create storage for AP
  // MatrixXd AP = A;

  // Create storage for s
  // VectorXd s = VectorXd::Zero(n);  // Initialize a vector s with zeros
  
  for(int i=0; i<n; ++i) s[i] = 0.0;
  
  int nrhs = 1;
  int info;
  
  // while (!R.empty() && wr.maxCoeff() > epsilon) {

  double wr_max = w[ R[0] ];
  for(int i=1; i<num_R; ++i) if(wr_max < w[ R[i] ]) wr_max = w[ R[i] ];

  double t1_ = omp_get_wtime();
  timer[2] += t1_ - t0_;
  
  while( (num_R > 0) && wr_max > epsilon) {
    double t2_ = omp_get_wtime();
    
    wwhile_count++;
    
    double max_dot_product = -std::numeric_limits<double>::infinity();

    int j_max = -1;

    //     for (int j : R) {
    //         double dot_product = w(j);
    //         if (dot_product > max_dot_product) {
    //             max_dot_product = dot_product;
    //             j_max = j;
    //         }
    //     }

    for(int j=0; j<num_R; ++j) {
      double dot_product = w[R[j]];
      //      printf(" -- j, w, max_dot_product, j_max= %i %f %f %i\n",R[j],w[R[j]],max_dot_product,j_max);
      if(dot_product > max_dot_product) {
	max_dot_product = dot_product;
	j_max = R[j];
      }
    }

    //     P.insert(j_max);  // Add the selected index to P

    P[num_P] = j_max;
    num_P++;
    
    //     R.erase(j_max);   // Remove the selected index from R
    
    int indx = 0;
    while(indx < num_R) {
      if(R[indx] == j_max) break;
      indx++;
    }
    for(int i=indx; i<num_R; ++i) R[i] = R[i+1];
    num_R--;

    double t7_ = omp_get_wtime();
    timer[7] += t7_ - t2_;
    
    //     //AP = AP.colwise().take(P);
    //     std::vector<int> pidx(P.begin(), P.end());
    //     //std::cout << " pidx size = " << pidx.size() << std::endl;
    //     // Construct the submatrix AP consisting of columns corresponding to indices in P
    //     AP = A(Eigen::placeholders::all, pidx);

    for(int i=0; i<num_P; ++i) {
      int indx = P[i];
      for(int j=0; j<num_rows; ++j) AP[j*num_P+i] = A[j*num_cols+indx];
    }

    double t8_ = omp_get_wtime();
    timer[8] += t8_ - t7_;
    
    //     VectorXd sP = (AP.transpose() * AP).ldlt().solve(AP.transpose() * y);  // Compute the least squares solution for the selected indices
    
    // APP = AP.transpose() * AP // (num_P x num_rows) * (num_rows x num_P) = num_P x num_P

    {
      const double alpha = 1.0;
      const double beta = 0.0;
      dgemm_((const char *) "N", (const char *) "T", &num_P, &num_P, &num_rows, &alpha, AP.data(), &num_P, AP.data(), &num_P, &beta, APP.data(), &num_P);
    }
    
#if 0
    if(num_P < 4) {
      printf("num_rows= %i  num_P= %i\n",num_rows,num_P);
      // printf("AP= \n");
      // for(int i=0; i<num_rows; ++i) {
      // 	for(int j=0; j<num_P; ++j) printf(" %f",AP[i*num_P+j]);
      // 	printf("\n");
      // }
      
      printf("APP(dgemm)= \n");
      for(int i=0; i<num_P; ++i) {
	for(int j=0; j<num_P; ++j) printf(" %f",APP[i*num_P+j]);
	printf("\n");
      }
      for(int i=0; i<num_P*num_P; ++i) APP[i] = -1.0;

#if 1
      for(int i=0; i<num_P; ++i) {
	for(int j=0; j<=i; ++j) {
	  double val = 0.0;
	  for(int k=0; k<num_rows; ++k) val += AP[k*num_P+i] * AP[k*num_P+j];
	  APP[i*num_P+j] = val;
	  APP[j*num_P+i] = val;
	}
      }
#else
      {    
	const double alpha = 1.0;
	const double beta = 0.0;
	dsyrk_((const char *) "U", (const char *) "T", &num_P, &num_rows, &alpha, AP.data(), &num_rows, &beta, APP.data(), &num_P);
      }
#endif
      
      // fill upper APP
      
      printf("APP(dsyrk)= \n");
      for(int i=0; i<num_P; ++i) {
	for(int j=0; j<num_P; ++j) printf(" %f",APP[i*num_P+j]);
	printf("\n");
      }
      
      for(int i=0; i<num_P-1; ++i)
	for(int j=i+1; j<num_P; ++j) APP[i*num_P+j] = APP[j*num_P+i];
      
      printf("APP(fill)= \n");
      for(int i=0; i<num_P; ++i) {
	for(int j=0; j<num_P; ++j) printf(" %f",APP[i*num_P+j]);
	printf("\n");
      }
      
      //if(num_P == 4) exit(1);
    }
#endif
    
    double t9_ = omp_get_wtime();
    timer[9] += t9_ - t8_;
    
    // APy = AP.tranpose() * y // (num_P x num_rows) * num_rows

    for(int i=0; i<num_P; ++i) {
      double val = 0.0;
      for(int j=0; j<num_rows; ++j) val += AP[j*num_P+i] * y[j];
      APy[i] = val;
    }

    double t6_ = omp_get_wtime();
    timer[10] += t6_ - t9_;
#if 1
    // dsytrf_((const char *) "U", &num_P, APP, &num_P, ipiv, work, &max_lwork, &info);
    // dsysv_((const char *) "U", &num_P, &nrhs, APP, &num_P, ipiv, APy, &num_P, work, &max_lwork, &info);

    dgetrf_(&num_P, &num_P, APP.data(), &num_P, ipiv.data(), &info);
    dgetrs_((char *) "N", &num_P, &nrhs, APP.data(), &num_P, ipiv.data(), APy.data(), &num_P, &info);
#else
    dgels_((const char *) "N", &num_P, &num_P, &nrhs, APP.data(), &num_P, APy.data(), &num_P, work.data(), &max_lwork, &info);
#endif
    
    // printf("num_P= %i  APy= ",num_P);
    // for(int i=0; i<num_P; ++i) printf(" %f",APy[i]);
    // printf("\n");

    //    if(num_P == 2) exit(1);
    
    double t10_ = omp_get_wtime();
    timer[11] += t10_ - t6_;
    
    for(int i=0; i<num_P; ++i) sP[i] = APy[i];

    //     s = VectorXd::Zero(n);  // Initialize a vector s with zeros
    //     int idx = 0;
    //     for (auto pi = P.begin(); pi != P.end(); pi++) 
    //     {
    //         s[*pi] = sP(idx);
    //         idx++;
    //     }

    for(int i=0; i<num_P; ++i) s[ P[i] ] = sP[i];

    double min_sP = sP[0];
    for(int j=1; j<num_P; ++j) if(sP[j] < min_sP) min_sP = sP[j];
    
    double t3_ = omp_get_wtime();
    timer[4] += t3_ - t2_;
    
    while(min_sP < zero) {
      swhile_count++;
      
      double alpha = std::numeric_limits<double>::infinity();  // Initialize alpha as positive infinity

      //         for (int i : P) {
      //             if (s(i) <= 0) {
      //                 double alpha_candidate = x(i) / (x(i) - s(i));
      //                 if (alpha_candidate < alpha) {
      //                     alpha = alpha_candidate;
      //                 }
      //             }
      //         }
      
      for(int i=0; i<num_P; ++i) {
	if(s[P[i]] < zero) {
	  double alpha_candidate = x[P[i]] / (x[P[i]] - s[P[i]]);
	  if(alpha_candidate < alpha) alpha = alpha_candidate;
	}
      }

      //         // Update the solution vector x with the computed alpha
      //         x += alpha * (s - x);

      for(int i=0; i<num_cols; ++i) x[i] += alpha * (s[i] - x[i]);
      
      //         // Move indices from P to R if their corresponding elements in x become non-positive
      //         for (auto it = P.begin(); it != P.end();) {
      //             int i = *it;
      //             if (x(i) <= 0.0) {
      //                 it = P.erase(it);
      //                 R.insert(i);
      //             } else {
      //                 ++it;
      //             }
      //         }

      //      printf("(reuse) num_P(before)= %i\n",num_P);
      
      int ii = 0;
      while(ii < num_P) {
	if(x[P[ii]] < zero) {
	  int ii_ = P[ii];
	  
	  for(int i=ii; i<num_P; ++i) P[i] = P[i+1];
	  num_P--;

	  R[num_R] = ii_;
	  
	  num_R++;
	} else ii++;
      }
      
      //printf("(reuse) num_P(after)= %i\n",num_P);
      
      //         // Recompute the submatrix AP and the least squares solution sP
      //         //AP = A;
      //         //AP = AP.colwise().take(P);
      //         std::vector<int> pidx(P.begin(), P.end());
      //         //AP = AP(pidx);
      //         AP = A(Eigen::placeholders::all, pidx); 

      for(int i=0; i<num_P; ++i) {
	int indx = P[i];
	for(int j=0; j<num_rows; ++j) AP[j*num_P+i] = A[j*num_cols+indx];
      }
      
      //         //s = VectorXd::Zero(n);
      //         s = VectorXd::Zero(n);  // Initialize a vector s with zeros
      
      for(int i=0; i<n; ++i) s[i] = 0.0;
      
      //         sP = (AP.transpose() * AP).ldlt().solve(AP.transpose() * y);
      
      // APP = AP.transpose() * AP // (num_P x num_rows) * (num_rows x num_P)

      {
	const double alpha = 1.0;
	const double beta = 0.0;
	dgemm_((const char *) "N", (const char *) "T", &num_P, &num_P, &num_rows, &alpha, AP.data(), &num_P, AP.data(), &num_P, &beta, APP.data(), &num_P);
      }
      
      // APy = AP.tranpose() * y

      {
	const double alpha = 1.0;
	const double beta = 0.0;
	const int inc = 1;
	//printf("dgemv_(r2) :: num_P= %i  num_rows= %i  alpha= %f  inc= %i, beta= %f\n",num_P, num_rows, alpha, inc, beta);
	dgemv_((const char *) "N", &num_P, &num_rows, &alpha, AP.data(), &num_P, y, &inc, &beta, APy.data(), &inc);
      }
#if 1
    // dsytrf_((const char *) "U", &num_P, APP, &num_P, ipiv, work, &max_lwork, &info);
    // dsysv_((const char *) "U", &num_P, &nrhs, APP, &num_P, ipiv, APy, &num_P, work, &max_lwork, &info);

      dgetrf_(&num_P, &num_P, APP.data(), &num_P, ipiv.data(), &info);
      dgetrs_((char *) "N", &num_P, &nrhs, APP.data(), &num_P, ipiv.data(), APy.data(), &num_P, &info);
#else
      dgels_((const char *) "N", &num_P, &num_P, &nrhs, APP.data(), &num_P, APy.data(), &num_P, work.data(), &max_lwork, &info);
#endif
      
      for(int i=0; i<num_P; ++i) sP[i] = APy[i];
    
      //         int idx = 0;
      //         for (auto pi = P.begin(); pi != P.end(); pi++) 
      //         {
      //             s[*pi] = sP(idx);
      //             idx++;
      //         }
      
      for(int i=0; i<num_P; ++i) s[ P[i] ] = sP[i];
      
      min_sP = sP[0];
      for(int j=1; j<num_P; ++j) if(sP[j] < min_sP) min_sP = sP[j];
    } // while(min_sP)
    
    double t4_ = omp_get_wtime();
    timer[5] += t4_ - t3_;
    
    //     x = s;  // Update the solution vector x with the non-negative least squares solution

    for(int i=0; i<n; ++i) x[i] = s[i];
    
    //     w = A.transpose() * (y - A * x);

    // A * x
    {
      const double alpha = 1.0;
      const double beta = 0.0;
      const int inc = 1;
      //printf("dgemv_(r3) :: num_cols= %i  num_rows= %i  alpha= %f  inc= %i, beta= %f\n",num_cols, num_rows, alpha, inc, beta);
      dgemv_((const char *) "T", &num_cols, &num_rows, &alpha, A, &num_cols, x, &inc, &beta, Ax.data(), &inc);
    }
    
    // A.transpose * (y-Ax)
    {
      for(int i=0; i<m; ++i) Ax[i] = y[i] - Ax[i];
	
      const double alpha = 1.0;
      const double beta = 0.0;
      const int inc = 1;
      //printf("dgemv_(r4) :: n= %i  m= %i  alpha= %f  inc= %i, beta= %f\n",n, m, alpha, inc, beta);
      dgemv_((const char *) "N", &n, &m, &alpha, A, &n, Ax.data(), &inc, &beta, w.data(), &inc);
    }
    
    //     wr.resize(R.size());

    //     idx = 0;
    //     for (int i : R) {
    //         wr(idx) = w(i);
    //         ++idx;
    //     }

    for(int i=0; i<num_R; ++i) wr[i] = w[R[i]];
    
    wr_max = w[ R[0] ];
    for(int i=1; i<num_R; ++i) if(wr_max < w[ R[i] ]) wr_max = w[ R[i] ];
    
    double t5_ = omp_get_wtime();
    timer[6] += t5_ - t4_;
    
    if(wwhile_count > 200) {
      printf("NNLS :: WARNING!!! wwhile_count > %i and breaking loop\n",wwhile_count);
      break;
    }
  } // while(numR && max_wr)

  timer[3] += omp_get_wtime() - t1_;
  
  printf("NNLS::solve(cpp_reuse) -- num_R= %i  num_P= %i  w_count= %i  s_count= %i\n",num_R,num_P,wwhile_count,swhile_count);
  
  for(int i=0; i<num_cols; ++i) x_old[i] = x[i];
  
  timer[0] += omp_get_wtime() - t0_;
}

int main() {
  // Example usage:
#if 0
  const int NUM_ROWS = 3;
  const int NUM_COLS = 2;
  double A[NUM_ROWS*NUM_COLS] = {1,2,3,4,5,6};
  double b[NUM_ROWS] = {7,8,9};
  double ref[NUM_COLS] = {0.0,1.78571429};
#endif

#if 1
  const int NUM_ROWS = 3;
  const int NUM_COLS = 2;
  double A[NUM_ROWS*NUM_COLS] = {1,2,3,4,5,6};
  double b[NUM_ROWS] = {5, 11, 17};
  double ref[NUM_COLS] = {1.0,2.0};
#endif
  
#if 0
  const int NUM_ROWS = 4;
  const int NUM_COLS = 3;
  double A[NUM_ROWS*NUM_COLS] = {1,2,3,4,5,6,7,8,9,10,11,12};
  double b[NUM_ROWS] = {13,14,15,16};
  double ref[NUM_COLS] = {0.0,0.0,1.66667};
#endif
  
#if 0
  const int NUM_ROWS = 5;
  const int NUM_COLS = 4;
  double A[NUM_ROWS*NUM_COLS] = {1,2,3,4,5,6,7,8,9,10,11,12, 13, 14, 15, 16, 17, 18, 19, 20};
  double b[NUM_ROWS] = {21, 22, 23, 24, 25};
  double ref[NUM_COLS] = {0.0,0.0,0.0,1.61364};
#endif
  
  printf("\nA(%i x %i)= \n", NUM_ROWS, NUM_COLS);
  for(int i=0; i<NUM_ROWS; ++i) {
    for(int j=0; j<NUM_COLS; ++j) printf(" %f",A[i*NUM_COLS + j]);
    printf("\n");
  }
  
  printf("\nb (%i)= \n",NUM_ROWS);
  for(int i=0; i<NUM_ROWS; ++i) printf(" %f",b[i]);
  printf("\n");
  
  double x[NUM_COLS];
  double eps = 1e-6;
  NNLS::non_negative_least_squares(A, b, x, NUM_ROWS, NUM_COLS, eps);

  printf("\nx(%i)= ",NUM_COLS);
  for(int i=0; i<NUM_COLS; ++i) printf(" %f",x[i]);

  printf("\nRef x= ");
  for(int i=0; i<NUM_COLS; ++i) printf(" %f",ref[i]);
  printf("\n");
}
