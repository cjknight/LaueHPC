
//#define EIGEN_USE_MKL_ALL
// Fed nnls_from_chatgpt.py to ChatGPT and ask for C++ version using Eigen
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <set>

//#define _DEBUG

using namespace Eigen;

VectorXd non_negative_least_squares(const MatrixXd& A, const VectorXd& y, double epsilon = 1e-6) {
    int m = A.rows();
    int n = A.cols();

    const double zero = 1e-12;
    
#ifdef _DEBUG
    std::cout << "num_rows(m)= " << m << " num_cols(n)= " << n << std::endl;
    
    {	
      std::cout << " -- A(" << A.size() << ")= ";
      //      for(int i=0; i<A.size(); ++i) std::cout << " " << A(i);
      std::cout << std::endl;
    }
#endif
    
    VectorXd x = VectorXd::Zero(n);  // Initialize the solution vector x with zeros
    std::set<int> P;  // Initialize the set P to store selected indices
    std::set<int> R;  // Initialize the set R with all indices
    for (int i = 0; i < n; ++i) {
        R.insert(i);
    }

    VectorXd w = A.transpose() * (y - A * x);
    VectorXd wr(R.size());

    //for (int i = 0; i < n; ++i) {
    //    wr(i) = w(R[i]);
    //}
    int idx = 0;
    for (auto ri = R.begin(); ri != R.end(); ri++) 
    {
        wr[idx] = w[*ri];
        idx++;
    }

    // Create storage for AP
    MatrixXd AP = A;

    
    // Create storage for s
    VectorXd s = VectorXd::Zero(n);  // Initialize a vector s with zeros

    while (!R.empty() && wr.maxCoeff() > epsilon) {
#ifdef _DEBUG
      std::cout << "Starting R while-loop w/ wr.maxCoeff= " << wr.maxCoeff() << std::endl;
#endif
      
      double max_dot_product = -std::numeric_limits<double>::infinity();
      int j_max = -1;
      
      for (int j : R) {
	double dot_product = w(j);
	//	std::cout << " -- j, w, max_dot_product, j_max= " << j << " " << w(j) << " " << max_dot_product << " " << j_max << std::endl;
	if (dot_product > max_dot_product) {
	  max_dot_product = dot_product;
	  j_max = j;
	}
      }
      
      P.insert(j_max);  // Add the selected index to P
      R.erase(j_max);   // Remove the selected index from R

#ifdef _DEBUG
      std::cout << " -- num_R= " << R.size() << " num_P= " << P.size() << " j_max= " << j_max << std::endl;
      
      std::cout << " -- P(" << P.size() << ")= ";
      for(auto it=P.begin(); it!=P.end(); ++it) std::cout << " " << *it;
      std::cout << std::endl;
#endif
      
      //AP = AP.colwise().take(P);
      std::vector<int> pidx(P.begin(), P.end());
      //std::cout << " pidx size = " << pidx.size() << std::endl;
      // Construct the submatrix AP consisting of columns corresponding to indices in P
      AP = A(Eigen::placeholders::all, pidx);

#ifdef _DEBUG
      std::cout << " -- AP(" << AP.size() << ")= ";
      //      for(int i=0; i<AP.size(); ++i) std::cout << " " << AP(i);
      std::cout << std::endl;
      
      {
	MatrixXd APP = AP.transpose() * AP;
	
	std::cout << " -- APP(" << APP.size() << ")= ";
	//	for(int i=0; i<APP.size(); ++i) std::cout << " " << APP(i);
	std::cout << std::endl;
      }
      
      {
	MatrixXd APy = AP.transpose() * y;
	
	std::cout << " -- APy(" << APy.size() << ")= ";
	//	for(int i=0; i<APy.size(); ++i) std::cout << " " << APy(i);
	std::cout << std::endl;
      }
#endif
      
      VectorXd sP = (AP.transpose() * AP).ldlt().solve(AP.transpose() * y);  // Compute the least squares solution for the selected indices

#ifdef _DEBUG
      std::cout << " -- sP(" << sP.size() << ")= ";
      for(int i=0; i<sP.size(); ++i) std::cout << " " << sP[i];
      std::cout << std::endl;
#endif
      
      //for (int i = 0; i < P.size(); ++i) {
      //    s(P[i]) = sP(i);
      //}
      s = VectorXd::Zero(n);  // Initialize a vector s with zeros
      int idx = 0;
      for (auto pi = P.begin(); pi != P.end(); pi++) 
        {
	  s[*pi] = sP(idx);
	  idx++;
        }
      
      //      while (sP.minCoeff() <= 0) {
      while (sP.minCoeff() < zero) {
#ifdef _DEBUG	
	std::cout << " -- Starting while() loop w/ min_sP= " << sP.minCoeff() << std::endl;
#endif
	
	double alpha = std::numeric_limits<double>::infinity();  // Initialize alpha as positive infinity
	
	for (int i : P) {
	  //	  if (s(i) <= 0) {
	  if (s(i) < zero) {
	    double alpha_candidate = x(i) / (x(i) - s(i));
	    if (alpha_candidate < alpha) {
	      alpha = alpha_candidate;
	    }
	  }
	}

#ifdef _DEBUG
	std::cout << " -- -- alpha= " << alpha << std::endl;

	{	
	  std::cout << " -- x(" << x.size() << ")= ";
	  for(int i=0; i<x.size(); ++i) std::cout << " " << x(i);
	  std::cout << std::endl;
	}
	
	std::cout << " -- num_R= " << R.size() << " num_P= " << P.size() << " j_max= " << j_max << std::endl;
#endif
	
	// Update the solution vector x with the computed alpha
	x += alpha * (s - x);
	
	// Move indices from P to R if their corresponding elements in x become non-positive
	for (auto it = P.begin(); it != P.end();) {
	  int i = *it;
	  //	  if (x(i) <= 0.0) {
	  if (x(i) < zero) {
	    it = P.erase(it);
	    R.insert(i);
	  } else {
	    ++it;
	  }
	}
	
#ifdef _DEBUG
	std::cout << " -- num_R= " << R.size() << " num_P= " << P.size() << " j_max= " << j_max << std::endl;
	
	std::cout << " -- P(" << P.size() << ")= ";
	for(auto it=P.begin(); it!=P.end(); ++it) std::cout << " " << *it;
	std::cout << std::endl;
	
	std::cout << " -- R(" << R.size() << ")= ";
	for(auto it=R.begin(); it!=R.end(); ++it) std::cout << " " << *it;
	std::cout << std::endl;
#endif
	
	// Recompute the submatrix AP and the least squares solution sP
	//AP = A;
	//AP = AP.colwise().take(P);
	std::vector<int> pidx(P.begin(), P.end());
	//AP = AP(pidx);
	AP = A(Eigen::placeholders::all, pidx);
	//s = VectorXd::Zero(n);
	s = VectorXd::Zero(n);  // Initialize a vector s with zeros
	sP = (AP.transpose() * AP).ldlt().solve(AP.transpose() * y);
	
	//for (int i = 0; i < P.size(); ++i) {
	//    s(P[i]) = sP(i);
	//}
	int idx = 0;
	for (auto pi = P.begin(); pi != P.end(); pi++) 
	  {
	    s[*pi] = sP(idx);
	    idx++;
	  }

#ifdef _DEBUG
	std::cout << " -- sP(" << sP.size() << ")= ";
	for(int i=0; i<sP.size(); ++i) std::cout << " " << sP[i];
	std::cout << std::endl;
	
	std::cout << " -- Finished while() loop w/ min_sP= " << sP.minCoeff() << std::endl;
#endif
      }
      
      x = s;  // Update the solution vector x with the non-negative least squares solution
      w = A.transpose() * (y - A * x);
      wr.resize(R.size());

#ifdef _DEBUG
      {	
	std::cout << " -- A(" << A.size() << ")= ";
	//for(int i=0; i<A.size(); ++i) std::cout << " " << A(i);
	std::cout << std::endl;
      }
      
      {
	MatrixXd At = A.transpose();
	
	std::cout << " -- At(" << At.size() << ")= ";
	//for(int i=0; i<At.size(); ++i) std::cout << " " << At(i);
	std::cout << std::endl;
      }
      
      {
	MatrixXd Ax = A * x;
	
	std::cout << " -- Ax(" << Ax.size() << ")= ";
	//for(int i=0; i<Ax.size(); ++i) std::cout << " " << Ax(i);
	std::cout << std::endl;
      }
      
      {	
	std::cout << " -- y(" << y.size() << ")= ";
	//for(int i=0; i<y.size(); ++i) std::cout << " " << y(i);
	std::cout << std::endl;
      }
      
      {	
	std::cout << " -- x(" << x.size() << ")= ";
	for(int i=0; i<x.size(); ++i) std::cout << " " << x(i);
	std::cout << std::endl;
      }
      
      {	
	std::cout << " -- w(" << w.size() << ")= ";
	for(int i=0; i<w.size(); ++i) std::cout << " " << w(i);
	std::cout << std::endl;
      }
#endif
      
      idx = 0;
      for (int i : R) {
	wr(idx) = w(i);
	++idx;
      }

#ifdef _DEBUG
      {	
	std::cout << " -- R.size= " << R.size() << "  wr(" << wr.size() << ")= ";
	for(int i=0; i<wr.size(); ++i) std::cout << " " << wr(i);
	std::cout << std::endl;
      }
      
      std::cout << "Finished R while-loop w/ wr.maxCoeff= ";
      if(wr.size() > 0) std::cout << wr.maxCoeff() << std::endl;
      else std::cout << std::endl;
#endif
    }
    
    return x;
}

VectorXf non_negative_least_squares_float(const MatrixXf& A, const VectorXf& y, double epsilon = 1e-6) {
    int m = A.rows();
    int n = A.cols();
    VectorXf x = VectorXf::Zero(n);  // Initialize the solution vector x with zeros
    std::set<int> P;  // Initialize the set P to store selected indices
    std::set<int> R;  // Initialize the set R with all indices
    for (int i = 0; i < n; ++i) {
        R.insert(i);
    }

    const float zero = 1e-8;
    
    VectorXf w = A.transpose() * (y - A * x);
    VectorXf wr(R.size());

    //for (int i = 0; i < n; ++i) {
    //    wr(i) = w(R[i]);
    //}
    int idx = 0;
    for (auto ri = R.begin(); ri != R.end(); ri++) 
    {
        wr[idx] = w[*ri];
        idx++;
    }

    while (!R.empty() && wr.maxCoeff() > epsilon) {
        double max_dot_product = -std::numeric_limits<double>::infinity();
        int j_max = -1;

        for (int j : R) {
            double dot_product = w(j);
            if (dot_product > max_dot_product) {
                max_dot_product = dot_product;
                j_max = j;
            }
        }

        P.insert(j_max);  // Add the selected index to P
        R.erase(j_max);   // Remove the selected index from R

        // Construct the submatrix AP consisting of columns corresponding to indices in P
        MatrixXf AP = A;
        //AP = AP.colwise().take(P);
        std::vector<int> pidx(P.begin(), P.end());
        //std::cout << " pidx size = " << pidx.size() << std::endl;
        AP = A(Eigen::placeholders::all, pidx);

        VectorXf s = VectorXf::Zero(n);  // Initialize a vector s with zeros
        VectorXf sP = (AP.transpose() * AP).ldlt().solve(AP.transpose() * y);  // Compute the least squares solution for the selected indices

        //for (int i = 0; i < P.size(); ++i) {
        //    s(P[i]) = sP(i);
        //}
        int idx = 0;
        for (auto pi = P.begin(); pi != P.end(); pi++) 
        {
            s[*pi] = sP(idx);
            idx++;
        }

	//        while (sP.minCoeff() <= 0) {
	while (sP.minCoeff() < zero) {
            double alpha = std::numeric_limits<double>::infinity();  // Initialize alpha as positive infinity

            for (int i : P) {
	      //                if (s(i) <= 0) {
	      if (s(i) < zero) {
                    double alpha_candidate = x(i) / (x(i) - s(i));
                    if (alpha_candidate < alpha) {
                        alpha = alpha_candidate;
                    }
                }
            }

            // Update the solution vector x with the computed alpha
            x += alpha * (s - x);

            // Move indices from P to R if their corresponding elements in x become non-positive
            for (auto it = P.begin(); it != P.end();) {
                int i = *it;
		//		if (x(i) <= 0.0) {
                if (x(i) < zero) {
                    it = P.erase(it);
                    R.insert(i);
                } else {
                    ++it;
                }
            }

            // Recompute the submatrix AP and the least squares solution sP
            AP = A;
            //AP = AP.colwise().take(P);
            std::vector<int> pidx(P.begin(), P.end());
            //AP = AP(pidx);
            AP = A(Eigen::placeholders::all, pidx);
            s = VectorXf::Zero(n);
            sP = (AP.transpose() * AP).ldlt().solve(AP.transpose() * y);

            //for (int i = 0; i < P.size(); ++i) {
            //    s(P[i]) = sP(i);
            //}
            int idx = 0;
            for (auto pi = P.begin(); pi != P.end(); pi++) 
            {
                s[*pi] = sP(idx);
                idx++;
            }
        }

        x = s;  // Update the solution vector x with the non-negative least squares solution
        w = A.transpose() * (y - A * x);
        wr.resize(R.size());

        idx = 0;
        for (int i : R) {
            wr(idx) = w(i);
            ++idx;
        }
    }

    return x;
}

int main() {
    // Example usage:
#if 0
    MatrixXd A(3, 2);
    A << 1, 2,
         3, 4,
         5, 6;

    VectorXd y(3);
    y << 7 , 8 , 9;
#endif
    
#if 1
    MatrixXd A(3, 2);
    A << 1, 2,
         3, 4,
         5, 6;

    VectorXd y(3);
    y << 5, 11, 17;
#endif
    
#if 0
    MatrixXd A(4, 3);
    A << 1, 2, 3,
         4, 5, 6, 
         7, 8, 9,
         10, 11, 12;

    VectorXd y(4);
    y << 13, 14, 15, 16;
#endif
    
#if 0
    MatrixXd A(5, 4);
    A << 1, 2, 3, 4,
         5, 6, 7, 8,
         9, 10, 11, 12,
         13, 14, 15, 16,
         17, 18, 19, 20;

    VectorXd y(5);
    y << 21, 22, 23, 24, 25;
#endif
    
    double eps = 1e-6;
    auto x = non_negative_least_squares(A, y, eps);
    std::cout << "x= " << x << std::endl;
}
