#!/bin/bash -l

export PYTHONPATH=${PWD}/..:$PYTHONPATH
echo "PYTHONPATH= ${PYTHONPATH}"

python test.py

