#!/bin/bash -l

# Assumes MKL has been loaded as a module

PYTHON_INC=`python -m pybind11 --includes`
PYTHON_LIB="-L/opt/local/Library/Frameworks/Python.framework/Versions/3.12/lib/ -lpython3.12"

#COMMAND="g++-mp-13 -O3 -g -std=c++11 ${PYTHON_INC} -c solver.cpp "

#echo "COMMAND= ${COMMAND}"
#${COMMAND}

#LAPACK=" -L/Users/cjknight/Documents/soft/lib/lapack/lib -llapack -lrefblas "
LAPACK=" -L/Users/cjknight/Documents/soft/lib/openblas/lib -lopenblas "

COMMAND="g++-mp-13 -O3 -fopenmp -g -std=c++11 -D_USE_PYBIND ${PYTHON_INC} -shared -fPIC -o my_nnls_solver.so solver.cpp ${LAPACK} -lpthread -lm -ldl -lgfortran ${PYTHON_LIB} "

echo "COMMAND= ${COMMAND}"
${COMMAND}

#-L /usr/lib/python3.10/config-3.10-x86_64-linux-gnu/ \
#-l python3.10

