#!/bin/sh

echo "For the output of Adaboost model"
python3 adaboost.py

echo "For the output of cleaned data"
python3 datacleaning.py

echo "For the output of random forest model"
python3 randomforest.py

echo "For the output of hist gradient boost model"
python3 histgradboost.py

echo "For the output of XGBoost model"
python3 xgboost.py

echo "For the output of LightGBM model"
python3 lgbm.py

echo "For the output of SVM model"
python3 svm.py