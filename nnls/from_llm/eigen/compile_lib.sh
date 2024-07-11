#!/bin/bash -l

# compile as library with Python binding

PYTHON_INC=`python -m pybind11 --includes`
PYTHON_LIB="-L/opt/local/Library/Frameworks/Python.framework/Versions/3.12/lib/ -lpython3.12"

echo "PYTHON_INC= ${PYTHON_INC}"

COMMAND="g++-mp-13 \
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
-L /Users/cjknight/Documents/soft/lib/lapack/lib -llapack -lrefblas  -lpthread -lm -ldl ${PYTHON_LIB} "

echo "COMMAND= ${COMMAND}"
${COMMAND}

#-L /usr/lib/python3.10/config-3.10-x86_64-linux-gnu/ \
#-l python3.10

