This assignment implements about SVM




There are 7 different implementations and the commands used for executing them are as follows:


1. python svmprimal.py


To run the SVM in the primal domain using stochastic sub-gradient descent with the first learning rate schedule.


2. python svmprimal2.py


To run the SVM in the primal domain using stochastic sub-gradient descent with the second learning rate schedule.


3. python svmprimaldifference.py


To compare the differences in weights, biases, training errors, and test errors between the two learning rate schedules for SVM in the primal domain.


4. python DualSVM.py


To run the SVM in the dual domain to compute weights and biases, and compare the results with the primal SVM.


5. python gausssvm.py


To implement the Gaussian kernel in the dual form for nonlinear SVM and report training and test errors for different hyperparameter combinations.


6. python supportvectorcalc.py


To calculate the number of support vectors for each combination of hyperparameters in the Gaussian kernel SVM and report overlaps between consecutive gamma values.


7. python kernelperceptron.py


To run the kernel Perceptron algorithm using the Gaussian kernel and compare its performance with the nonlinear SVM.