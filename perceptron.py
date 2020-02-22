# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import random
import sys

'''
    This program implements a binary classifier using the perceptron algorithm. It can classify the iris dataset in
    three different ways:
        - Setosa Vs non-Setosa data points
        - Versicolor Vs non-Versicolor data points
        - Virginica Vs non-Virginica data points

    The Perceptron Algorithm:
    1. Start with a random set of weights w.
    2. Pick an arbitrary pattern x ∈ C1 ∪ C2.
    3. If x ∈ C1 and x · w < 0 goto 2. Else w → w − λx. Goto 2
    4. If x ∈ C2 and x · w ≥ 0 goto 2. Else w → w + λx. Goto 2
'''

# Implements the steps 3 and 4 of the Perceptron algorithm that adjust the weights


def adjust_weights(x, weights, classes, learning_rate):
    # multiply the vector of inputs with the vector of weights plus the weight of the bias
    dot_product = np.dot(x, weights[:4]) + weights[4]

    # Perceptron algorithm step 3.
    # If x ∈ C1 and dot_product < 0 goto 2. Else weights = weights − x. Goto 2
    if classes[i] == 0:
        if dot_product >= 0:
            weights[:4] -= learning_rate * x
            # bias weight
            weights[4] -= learning_rate

    # Perceptron algorithm step 4.
    # If x ∈ C2 and dot_product ≥ 0 goto 2. Else weights = weights + x. Goto 2
    else:  # classes[i] == 'non-'
        if dot_product < 0:
            weights[:4] += learning_rate * x
            # bias weight
            weights[4] += learning_rate

    return weights


# Calculates the dot product of the input and weights plus the bias and uses this to predict the class of the input data point
def predict_class(x, weights):
    # multiply the vector of inputs with the vector of weights plus the weight of the bias
    # (bias = 1 so it's not necessary to multiply it with 1)
    if weights.size == 5:  # hidden layer
        dot_product = np.dot(x, weights[:4]) + weights[4]
    else:  # output layer
        dot_product = np.dot(x, weights[:2]) + weights[2]
    if dot_product >= 0:
        return 1
    else:
        return 0


# Parse command line arguments
if(len(sys.argv) > 1):
    classification = sys.argv[1]

if(len(sys.argv) != 2 or (classification != "setosa" and classification != "versicolor" and classification != "virginica")):
    print("\npy perceptron.py <class>\nWhere class can be:\n- setosa (for Setosa Vs non-Setosa classification)\n- "
    "versicolor (for Versicolor Vs non-Versicolor classification)\n- virginica (for Virginica Vs non-Virginica "
    "classification)\n\n")
    sys.exit(0)

print("\nClassifying ", classification, " Vs non-", classification, sep="")

# Read the input file
df = pd.read_csv("./data/iris.data", header=None)

# Four Inputs: sepal length, sepal width, petal length, petal width
inputs = df.iloc[0:150, [0, 1, 2, 3]].values

# Three classes: Iris-setosa, Iris-versicolor, Iris-virginica
original_classes = df.iloc[0:150, 4].values

# Convert the three Classes into two classes
classes = np.where(original_classes == "Iris-" + classification, 1, 0)
classes_string = np.where(original_classes == "Iris-" +
                          classification, classification, "non-" + classification)


# ========================== Perceptron ==========================


# The output of the perceptron
outputs = np.zeros(150)

# Perceptron algorithm step 1.
# Generate random vector of weights. Five weights: Four weights for the four inputs plus one for the bias, initialized
# with random numbers [0, 1)
weights = np.random.rand(5)

# Iterations -> how many times the four steps of the perceptron algorithm are repeated

iterations = 0
max_iterations = 100000

# Number of consequent iterations of non-changing weights
convergence_count = 0

# Maximum number of consequent iterations of non-changing weights
max_convergence_count = 1500

# Iterate a max of max_iterations iterations or stop after the maximum number of consequent iterations of non-changing
# weights which means the algorithm converged
while convergence_count < max_convergence_count and iterations < max_iterations:

    # Perceptron algorithm step 2.
    # Pick a random x from either of the two classes
    i = random.randint(0, 149)
    iterations += 1
    prev_weights = np.copy(weights)

    # Perceptron algorithm step 3 and 4
    weights = adjust_weights(
        x=inputs[i], weights=weights, classes=classes, learning_rate=0.01)

    # if the weights didn't change from the previous iteration
    if np.array_equal(weights, prev_weights):
        convergence_count += 1
    else:
        convergence_count = 0

if iterations == max_iterations:
    print("The algorithm didn't converge")
else:
    print("The algorithm converged")

print("Number of iterations: ", iterations, "\n")

# Use the trained perceptron to predict the class (e.g. setosa=1 or non-setosa=0) which is the output of this perceptron
for i in range(0, 150):
    outputs[i] = predict_class(inputs[i], weights)

# Verify the correctness of the predicted class of each point in the dataset and count the misclassified points
misclassified = 0
for i in range(0, 150):
    if outputs[i] == 1:
        myClass = classification
    else:
        myClass = "non-" + classification

    if myClass != classes_string[i]:

        print("Misclassified Point: ", inputs[i], ". Actual class is ", original_classes[i],
              ", but classified as", myClass)
        misclassified += 1

if misclassified == 0:
    print("All points classified correctly!\n")
else:
    print("\nTotal misclassified points: ", misclassified, "\n")
