#!/bin/sh

echo "For accuracy score of back propagation"
python3 backpropagation.py

echo "For the results of stochastic gradient descent"
python3 sgd.py

echo "For the results of stochastic gradient descent by initializing weights with 0"
python3 sgdzero.py

echo "For the results of activation functions"
python3 activation.py