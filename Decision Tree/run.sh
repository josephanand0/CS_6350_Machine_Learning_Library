#!/bin/sh

echo "Average prediction errors for car dataset"
python3 decisiontreeforcar.py

echo "Average prediction errors for bank dataset -unknown as particular attribute value"
python3 decisiontreeforbank.py

echo "Average prediction errors for bank dataset -unknown as attribute value missing"
python3 decisiontreeforbankmissing.py