#!/bin/bash -l

# Assumes MKL has been loaded as a module

PYTHON_INC=`python -m pybind11 --includes`

#LAPACK=" -L/Users/cjknight/Documents/soft/lib/lapack/lib -llapack -lrefblas "
LAPACK=" -L/Users/cjknight/Documents/soft/lib/openblas/lib -lopenblas "

COMMAND="g++-mp-13 -O3 -fopenmp -g -std=c++11 ${PYTHON_INC} solver.cpp ${LAPACK} -lpthread -lm -ldl -lgfortran "
#COMMAND="g++-mp-13 -O3 -g -std=c++11 ${PYTHON_INC} -shared -fPIC -o solver.so solver.cpp ${LAPACK} -lpthread -lm -ldl -lgfortran "

echo "COMMAND= ${COMMAND}"
${COMMAND}

#-L /usr/lib/python3.10/config-3.10-x86_64-linux-gnu/ \
#-l python3.10

