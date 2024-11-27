#!/bin/sh

echo "For the output of 2a"
python svmprimal.py

echo "For the output of 2b"
python svmprimal2.py

echo "For the output of 2c"
python svmprimaldifference.py

echo "For the output of 3a"
python DualSVM.py

echo "For the output of 3b"
python gausssvm.py

echo "For the output of 3c"
python supportvectorcalc.py

echo "For the output of 3d"
python kernelperceptron.py