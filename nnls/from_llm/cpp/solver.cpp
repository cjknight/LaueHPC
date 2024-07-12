#include <iostream>
#include <vector>
#include <set>
#include <limits>

#include "solver.h"

//#define _DEBUG

extern "C" {
  void dgels_(const char * trans, const int * m, const int * n, const int * nrhs,
	      double * A, const int * lda, double * B, const int * ldb, double * work,                     
	      int * lwork, int * info);

  void dgemv_(const char * trans, const int * m, const int * n, const double * alpha,
	      const double * a, const int * lda, const double * x, const int * incx,
             const double * beta, double * y, const int * incy);
}

using namespace NNLS;

void NNLS::init(int nr, int nc)
{
#ifdef _DEBUG
  printf("Calling non_negative_least_squares::init()\n");
#endif

  int maxv = (nc > nr) ? nc : nr;
  if(maxv > max_size_vector) {
    max_size_vector = maxv + 100;

    if(R) free(R);
    R = (int *) malloc(max_size_vector * sizeof(int));
    
    if(P) free(P);
    P = (int *) malloc(max_size_vector * sizeof(int));

    if(w) free(w);
    w = (double *) malloc(max_size_vector * sizeof(double));
    
    if(wr) free(wr);
    wr = (double *) malloc(max_size_vector * sizeof(double));
    
    if(s) free(s);
    s = (double *) malloc(max_size_vector * sizeof(double));
    
    if(sP) free(sP);
    sP = (double *) malloc(max_size_vector * sizeof(double));
    
    if(APy) free(APy);
    APy = (double *) malloc(max_size_vector * sizeof(double));
    
    if(Ax) free(Ax);
    Ax = (double *) malloc(max_size_vector * sizeof(double));
  }

  if(maxv*maxv > max_size_matrix) {
    max_size_matrix = maxv*maxv + 100;
    
    if(AP) free(AP);
    AP = (double *) malloc(max_size_matrix * sizeof(double));
    
    if(APP) free(APP);
    APP = (double *) malloc(max_size_matrix * sizeof(double));
  }
  
  initialized = true;
}

void NNLS::finalize()
{
  if(R) free(R);
  if(P) free(P);

  if(w) free(w);
  if(wr) free(wr);
  if(s) free(s);
  if(sP) free(sP);
  if(APy) free(APy);
  
  if(Ax) free(Ax);
  if(AP) free(AP);
  if(APP) free(APP);
}

void NNLS::non_negative_least_squares(double * A, double * y, double * x, int num_rows, int num_cols, double epsilon)
{ 
  //  if(!initialized)
  NNLS::init(num_rows, num_cols);
  
  // int m = A.rows();
  // int n = A.cols();
  int m = num_rows;
  int n = num_cols;

  const double zero = 1e-15;
  
  int wwhile_count = 0;
  int swhile_count = 0;
  
#if defined(_DEBUG)
  printf("num_rows(m)= %i  num_cols(n)= %i\n",num_rows,num_cols);
  
  {	
    printf(" -- A(%i)= ", num_rows*num_cols);
    //    for(int i=0; i<num_rows*num_cols; ++i) printf(" %f", A[i]);
    printf("\n");
  }
#endif
  
  // VectorXd x = VectorXd::Zero(n);  // Initialize the solution vector x with zeros

  for(int i=0; i<num_cols; ++i) x[i] = 0.0;

  //    std::set<int> R;  // Initialize the set R with all indices
  //    for (int i = 0; i < n; ++i) {
  //        R.insert(i);
  //    }
  
  int num_R = n;
  for(int i=0; i<num_R; ++i) R[i] = i;
  
  //    std::set<int> P;  // Initialize the set P to store selected indices
  
  int num_P = 0;
  
  // VectorXd w = A.transpose() * (y - A * x);
  // x = 0, so x = A.transpose() * y
  
  // -- +++++++++++++++++++++++++++++++++++++

  // A.transpose * y
  
  {
    const double alpha = 1.0;
    const double beta = 0.0;
    const int inc = 1;
    dgemv_((const char *) "T", &num_cols, &num_rows, &alpha, A, &num_cols, y, &inc, &beta, w, &inc);
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
  int lwork = -1;
  int info;

#ifdef _DEBUG
  printf("about to call dgels_()\n");
#endif

  double * work = (double *) malloc(sizeof(double));
  
  dgels_((const char *) "N", &num_rows, &num_rows, &nrhs, APP, &num_rows, APy, &num_rows, work, &lwork, &info);

#ifdef _DEBUG
  printf("lwork= %i  work[0]= %f  info= %i\n",lwork,work[0],info);
#endif
  
  lwork = static_cast<int>(work[0] + 0.5);

#ifdef _DEBUG
  printf("updated value of lwork= %i\n\n",lwork);
#endif
  
  free(work);
  work = (double *) malloc(lwork * sizeof(double));
  
  // while (!R.empty() && wr.maxCoeff() > epsilon) {

  double wr_max = w[ R[0] ];
  for(int i=1; i<num_R; ++i) if(wr_max < w[ R[i] ]) wr_max = w[ R[i] ];
  
  while( (num_R > 0) && wr_max > epsilon) {
#ifdef _DEBUG
    printf("Starting R while-loop w/ wr.maxCoeff= %f\n",wr_max);
#endif

    wwhile_count++;
    
    //     double max_dot_product = -std::numeric_limits<double>::infinity();
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

#if 1 // does it need to be ordered?? I don't think so 
    P[num_P] = j_max;
#else
    {
      int i;
      for(i=0; i<num_P; ++i) if(j_max < P[i]) break;
      for(int j=num_P; j>i; --j) P[j] = P[j-1];
      P[i] = j_max;
    }
#endif
    num_P++;
    
    //     R.erase(j_max);   // Remove the selected index from R
    
    int indx = 0;
    while(indx < num_R) {
      if(R[indx] == j_max) break;
      indx++;
    }
    for(int i=indx; i<num_R; ++i) R[i] = R[i+1];
    num_R--;

#ifdef _DEBUG
    printf(" -- num_R= %i  num_P= %i  j_max= %i\n",num_R, num_P, j_max);
    
    printf(" -- P(%i)= ",num_P);
    for(int i=0; i<num_P; ++i) printf(" %i",P[i]);
    printf("\n");
#endif
    
    //     //AP = AP.colwise().take(P);
    //     std::vector<int> pidx(P.begin(), P.end());
    //     //std::cout << " pidx size = " << pidx.size() << std::endl;
    //     // Construct the submatrix AP consisting of columns corresponding to indices in P
    //     AP = A(Eigen::placeholders::all, pidx);

    for(int i=0; i<num_P; ++i) {
      int indx = P[i];
      for(int j=0; j<num_rows; ++j) {
#ifdef _DEBUG
	//	printf(" -- ij= %i %i  indx1= %i  indx2= %i  A= %f\n",i,j,j*num_cols+i,j*num_cols+indx,A[j*num_cols+indx]);
#endif
	AP[j*num_P+i] = A[j*num_cols+indx];
      }
    }

#ifdef _DEBUG
    //printf(" -- num_P= %i  num_rows= %i\n",num_P,num_rows);
    printf(" -- AP(%i)= ",num_P*num_rows);
    //    for(int i=0; i<num_P*num_rows; ++i) printf(" %f",AP[i]);
    printf("\n");
#endif
    
    //     VectorXd sP = (AP.transpose() * AP).ldlt().solve(AP.transpose() * y);  // Compute the least squares solution for the selected indices

#ifdef _DEBUG
    // printf(" -- num_P= %i  num_rows= %i\n",num_P,num_rows);
#endif
    
    // APP = AP.transpose() * AP // (num_P x num_rows) * (num_rows x num_P) = num_P x num_P

    for(int i=0; i<num_P; ++i)
      for(int j=0; j<num_P; ++j) {

	double val = 0.0;
	for(int k=0; k<num_rows; ++k) {
	  val += AP[k*num_P+i] * AP[k*num_P+j];
#ifdef _DEBUG
	  //printf(" -- AP^T.AP :: ij= %i %i  k= %i  APt= %f  AP= %f  val= %f\n",i,j,k,AP[k*num_P+i],AP[k*num_P+j],val);
#endif
	}
	APP[i*num_P+j] = val;
      }

#ifdef _DEBUG
    printf(" -- APP(%i)= ",num_P*num_P);
    //    for(int i=0; i<num_P*num_P; ++i) printf(" %f",APP[i]);
    printf("\n");
#endif
    
    // APy = AP.tranpose() * y // (num_P x num_rows) * num_rows

    for(int i=0; i<num_P; ++i) {
      double val = 0.0;
      for(int j=0; j<num_rows; ++j) val += AP[j*num_P+i] * y[j];
      APy[i] = val;
    }

#ifdef _DEBUG
    printf(" -- APy(%i)= ",num_P);
    //    for(int i=0; i<num_P; ++i) printf(" %f",APy[i]);
    printf("\n");
    
    //    printf("calling initial dgels_() w/ lwork= %i\n",lwork);
#endif
    
    dgels_((const char *) "N", &num_P, &num_P, &nrhs, APP, &num_P, APy, &num_P, work, &lwork, &info);    

    for(int i=0; i<num_P; ++i) sP[i] = APy[i];

#ifdef _DEBUG
    printf(" -- sP(%i)= ",num_P);
    for(int i=0; i<num_P; ++i) printf(" %f",sP[i]);
    printf("\n");
#endif
    
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
    
    //     while (sP.minCoeff() <= 0) {
    while(min_sP < zero) {
#ifdef _DEBUG
      printf(" -- Starting while() loop w/ min_sP= %f\n",min_sP);
#endif

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
	  //	  printf(" -- -- i= %i  x= %f  P= %i  s= %f  alpha_candidate= %f  alpha= %f\n",i,x[i],P[i],s[P[i]],alpha_candidate,alpha);
	  if(alpha_candidate < alpha) alpha = alpha_candidate;
	}
      }

#ifdef _DEBUG
      printf(" -- -- alpha= %f\n",alpha);
#endif

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


#ifdef _DEBUG
      {	
	printf(" -- x(%i)= ", num_cols);
	for(int i=0; i<num_cols; ++i) printf(" %f", x[i]);
	printf("\n");
      }
#endif
      
#ifdef _DEBUG
      printf(" -- num_R= %i  num_P= %i\n",num_R, num_P);
      
      printf(" -- P(%i)= ",num_P);
      for(int i=0; i<num_P; ++i) printf(" %i",P[i]);
      printf("\n");
      
      printf(" -- R(%i)= ",num_R);
      for(int i=0; i<num_R; ++i) printf(" %i",R[i]);
      printf("\n");
#endif
      
      //      printf("Shifting P to R...\n");
      int ii = 0;
      while(ii < num_P) {
	//	printf(" -- ii= %i  P= %i  x= %f\n",ii,P[ii],x[P[ii]]);
	if(x[P[ii]] < zero) {
	  int ii_ = P[ii];
	  
	  for(int i=ii; i<num_P; ++i) P[i] = P[i+1];
	  num_P--;

#if 1
	  R[num_R] = ii_;
#else
	  {
	    int i;
	    for(i=0; i<num_R; ++i) if(ii_ < R[i]) break;
	    for(int j=num_R; j>i; --j) R[j] = R[j-1];
	    R[i] = ii_;
	  }
#endif
	  num_R++;
	} else ii++;
      }
      //      printf(" -- finished\n");
      
#ifdef _DEBUG
      printf(" -- num_R= %i  num_P= %i\n",num_R, num_P);
      
      printf(" -- P(%i)= ",num_P);
      for(int i=0; i<num_P; ++i) printf(" %i",P[i]);
      printf("\n");
      
      printf(" -- R(%i)= ",num_R);
      for(int i=0; i<num_R; ++i) printf(" %i",R[i]);
      printf("\n");
#endif
      
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
      
      //      printf(" -- Computing APP num_rows= %i  num_P= %i\n",num_rows,num_P);
      
      // APP = AP.transpose() * AP // (num_P x num_rows) * (num_rows x num_P)
      
      //      printf(" -- Computing APP\n");
      for(int i=0; i<num_P; ++i)
	for(int j=0; j<num_P; ++j) {
	  double val = 0.0;
	  for(int k=0; k<num_rows; ++k) val += AP[k*num_P+i] * AP[k*num_P+j];
	  APP[i*num_P+j] = val;
	}
      
      // APy = AP.tranpose() * y
      
      //      printf(" -- Computing APy\n");
      for(int i=0; i<num_P; ++i) {
	double val = 0.0;
	for(int j=0; j<num_rows; ++j) val += AP[j*num_P+i] * y[j];
	APy[i] = val;
      }

      //      printf("about to call dgels_()\n");
      dgels_((const char *) "N", &num_P, &num_P, &nrhs, APP, &num_P, APy, &num_P, work, &lwork, &info);
      //      printf("  -- finished.\n");
      
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

#ifdef _DEBUG
      printf(" -- Finished while() loop w/ min_sP= %f\n",min_sP);
#endif
    } // while(min_sP)
    
    //     x = s;  // Update the solution vector x with the non-negative least squares solution

    for(int i=0; i<n; ++i) x[i] = s[i];
    
    //     w = A.transpose() * (y - A * x);

    // A * x
    
    for(int i=0; i<num_rows; ++i) {
      double val = 0.0;
      for(int j=0; j<num_cols; ++j) val += A[i*num_cols + j] * x[j];
      Ax[i] = val;
    }
    
    // A.transpose * (y-Ax)
    
    for(int i=0; i<n; ++i) {
      double val = 0.0;
      for(int j=0; j<m; ++j) val += A[j*n+i] * (y[j] - Ax[j]);
      w[i] = val;
    }
    
    //     wr.resize(R.size());

    //     idx = 0;
    //     for (int i : R) {
    //         wr(idx) = w(i);
    //         ++idx;
    //     }

    //    for(int i=0; i<num_R; ++i) wr[ R[i] ] = w[i];
    for(int i=0; i<num_R; ++i) wr[i] = w[R[i]];
    
    wr_max = w[ R[0] ];
    for(int i=1; i<num_R; ++i) if(wr_max < w[ R[i] ]) wr_max = w[ R[i] ];

#ifdef _DEBUG
    {	
      printf(" -- A(%i)= ", num_rows*num_cols);
      //for(int i=0; i<num_rows*num_cols; ++i) printf(" %f", A[i]);
      printf("\n");
    }
    
    {	
      printf(" -- Ax(%i)= ", num_rows);
      //for(int i=0; i<num_rows; ++i) printf(" %f", Ax[i]);
      printf("\n");
    }
    
    {	
      printf(" -- y(%i)= ", m);
      //for(int i=0; i<m; ++i) printf(" %f", y[i]);
      printf("\n");
    }
    
    {	
      printf(" -- x(%i)= ", num_cols);
      for(int i=0; i<num_cols; ++i) printf(" %f", x[i]);
      printf("\n");
    }
    
    {	
      printf(" -- w(%i)= ", num_cols);
      for(int i=0; i<num_cols; ++i) printf(" %f", w[i]);
      printf("\n");
    }

    {	
      printf(" -- R.size= %i  wr(%i)= ", num_R, num_R);
      for(int i=0; i<num_R; ++i) printf(" %f", wr[i]);
      printf("\n");
    }
    
    printf("Finished R while-loop w/ wr.maxCoeff= ");
    if(num_R > 0) printf("%f\n",wr_max);
    else printf("\n");

    if(wwhile_count > 200) {
      printf("NNLS :: WARNING!!! wwhile_cout > %i and breaking loop\n",wwhile_count);
      break;
    }
#endif
  } // while(numR && max_wr)

  printf("NNLS::solve -- w_count= %i  s_count= %i\n",wwhile_count,swhile_count);
  
  free(work);
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
