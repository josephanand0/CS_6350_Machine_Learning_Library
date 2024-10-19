#!/bin/sh

echo "For results of Adaboost"
python3 Adaboostbank.py

echo "For results of  bagged tree"
python3 BaggedTrees.py

echo "For results of biasvariance"
python3 biasvariance.py

echo "For results of randomforest"
python3 randomforest.py