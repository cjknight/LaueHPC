#!/bin/bash

# compile as stand-alone app

#LAPACK=" -L/Users/cjknight/Documents/soft/lib/lapack/lib -llapack -lrefblas "
LAPACK=" -L/Users/cjknight/Documents/soft/lib/openblas/lib -lopenblas "

COMMAND="g++-mp-13 \
-O3 \
-g \
-D EIGEN_DONT_PARALLELIZE \
-I /Users/cjknight/Documents/soft/lib/eigen/include \
-I /Users/cjknight/Documents/soft/lib/eigen/unsupported/include \
-std=c++14 \
from_chatgpt2.cpp  \
${LAPACK} -lpthread -lm -ldl "

echo "COMMAND= ${COMMAND}"
${COMMAND}

exit

# compile as library with Python binding

PYTHON_INC=`python3 -m pybind11 --includes`

COMMAND="g++-mp-12 \
-O3 \
-g \
-D EIGEN_DONT_PARALLELIZE \
-I /Users/cjknight/Documents/soft/lib/eigen/include \
-I /Users/cjknight/Documents/soft/lib/eigen/unsupported/include \
-std=c++14 ${PYTHON_INC} \
-shared \
-fPIC \
-o solver.so \
pyeigen1.cpp \
from_chatgpt2.cpp  \
${LAPACK} -lpthread -lm -ldl "

echo "COMMAND= ${COMMAND}"
${COMMAND}

#-L /usr/lib/python3.10/config-3.10-x86_64-linux-gnu/ \
#-l python3.10

