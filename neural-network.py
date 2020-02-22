# -*- coding: utf-8 -*-
"""
================================================================================
                                .-~~-.--.
                               :         )
                         .~ ~ -.\       /.- ~~ .
                         >       `.   .'       <
                        (         .- -.         )
                         `- -.-~  `- -'  ~-.- -'
                           (        :        )           _ _ .-:
                            ~--.    :    .--~        .-~  .-~  }
                                ~-.-^-.-~ \_      .~  .-~   .~
                                         \ \'     \ '_ _ -~
                                          `.`.    //
                                 . - ~ ~-.__`.`-.//
                             .-~   . - ~  }~ ~ ~-.~-.
                           .' .-~      .-~       :/~-.~-./:
                          /_~_ _ . - ~                 ~-.~-._
                                                           ~-.<
                                                           
================================================================================

"""

import numpy as np
import pandas as pd
import random


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
    # (bias = 1 so it"s not necessary to multiply it with 1)
    if weights.size == 5:  # hidden layer
        dot_product = np.dot(x, weights[:4]) + weights[4]
    else:  # output layer
        dot_product = np.dot(x, weights[:2]) + weights[2]
    if dot_product >= 0:
        return 1
    else:
        return 0


# Calculates the accuracy of the neural network on the Iris dataset
def calculate_accuracy(original_classes, setosa_outputs, virginica_outputs, versicolor_outputs):
    # Calculate Accuracy
    misclassified = 0
    for i in range(0, 150):
        if versicolor_outputs[i] == 1 and setosa_outputs[i] == 0 and \
                virginica_outputs[
                    i] == 0 and original_classes[i] != "Iris-versicolor":
            misclassified += 1
        elif versicolor_outputs[i] == 0 and setosa_outputs[i] == 1 and \
                virginica_outputs[i] == 0 and original_classes[i] != "Iris-setosa":
            misclassified += 1
        elif versicolor_outputs[i] == 0 and setosa_outputs[i] == 0 and \
                virginica_outputs[i] == 1 and original_classes[i] != "Iris-virginica":
            misclassified += 1

    if misclassified == 0:
        print("All points classified correctly!")
    else:
        print("\nTotal misclassified points: ", misclassified)

    accuracy = (150 - misclassified) * 100 / 150
    print("Accuracy: ", accuracy, "%\n")


def printClassification(original_classes, setosa_outputs, virginica_outputs, versicolor_outputs):
    print("Classification of the Iris dataset:\n")
    print("Data Point\tActual Class\t\tPredicted Class")
    for i in range(0, 150):
        
        print(i, original_classes[i], sep="\t\t", end="\t\t")
        if versicolor_outputs[i] == 1 and setosa_outputs[i] == 0 and virginica_outputs[i] == 0:
            print("Iris-versicolor")
        elif versicolor_outputs[i] == 0 and setosa_outputs[i] == 1 and virginica_outputs[i] == 0:
            print("Iris-setosa")
        elif versicolor_outputs[i] == 0 and setosa_outputs[i] == 0 and virginica_outputs[i] == 1:
            print("Iris-virginica")


print(__doc__)

# Read the input file
df = pd.read_csv("./data/iris.data", header=None)

# Four Inputs: sepal length, sepal width, petal length, petal width
inputs = df.iloc[0:150, [0, 1, 2, 3]].values

# Three classes: Iris-setosa, Iris-versicolor, Iris-virginica
original_classes = df.iloc[0:150, 4].values

# ========================== Hidden Layer: Iris-setosa Perceptron ==========================

# Convert the three Classes into two classes for the setosa vs non-setosa classification
classes_setosa = np.where(original_classes == "Iris-setosa", 1, 0)

# The output of the setosa perceptron
hidden_setosa_outputs = np.zeros(150)

# Perceptron algorithm step 1
# Generate random vector of weights. Five weights: Four weights for the four inputs plus one for the bias, initialized
# with random numbers [0, 1)
hidden_setosa_weights = np.random.rand(5)

# Iterations -> how many times the four steps of the perceptron algorithm are repeated
iterations = 0
max_iterations = 1500

# Number of consequent iterations of non-changing weights
convergence_count = 0

# Maximum number of consequent iterations of non-changing weights
max_convergence_count = 300

# Iterate a max of max_iterations iterations or stop after the maximum number of consequent iterations of non-changing
# weights which means the algorithm converged
while convergence_count < max_convergence_count and iterations < max_iterations:

    # Perceptron algorithm step 2
    # Pick a random x from either of the two classes
    i = random.randint(0, 149)
    iterations += 1
    prev_hidden_setosa_weights = np.copy(hidden_setosa_weights)

    # Perceptron algorithm step 3 and 4
    hidden_setosa_weights = adjust_weights(
        x=inputs[i], weights=hidden_setosa_weights, classes=classes_setosa, learning_rate=1)

    # if the weights didn't change from the previous iteration
    if np.array_equal(hidden_setosa_weights, prev_hidden_setosa_weights):
        convergence_count += 1
    else:
        convergence_count = 0

# Use the trained perceptron to predict the class (setosa=1 or non-setosa=0) which is the output of this hidden layer
# neuron
for i in range(0, 150):
    hidden_setosa_outputs[i] = predict_class(inputs[i], hidden_setosa_weights)

# ========================== Hidden Layer: Iris-virginica Perceptron ==========================

# Convert the three Classes into two classes for the virginica vs non-virginica classification
classes_virginica = np.where(original_classes == "Iris-virginica", 1, 0)

# The output of the virginica perceptron
hidden_virginica_outputs = np.zeros(150)

# Perceptron algorithm step 1
# Generate random vector of weights. Five weights: Four weights for the four inputs plus one for the bias, initialized
# with random numbers [0, 1)
hidden_virginica_weights = np.random.rand(5)

# Iterations -> how many times the four steps of the perceptron algorithm are repeated
iterations = 0
max_iterations = 15000

# Number of consequent iterations of non-changing weights
convergence_count = 0

# Maximum number of consequent iterations of non-changing weights
max_convergence_count = 1500

# Iterate a max of max_iterations iterations or stop after the maximum number of consequent iterations of non-changing
# hidden_virginica_weights which means the algorithm converged
while convergence_count < max_convergence_count and iterations < max_iterations:

    # Perceptron algorithm step 2
    # Pick a random x from either of the two classes
    i = random.randint(0, 149)
    iterations += 1
    prev_hidden_virginica_weights = np.copy(hidden_virginica_weights)

    # Perceptron algorithm step 3 and 4
    hidden_virginica_weights = adjust_weights(x=inputs[i], weights=hidden_virginica_weights, classes=classes_virginica,
                                              learning_rate=0.01)

    # if the weights didn't change from the previous iteration
    if np.array_equal(hidden_virginica_weights, prev_hidden_virginica_weights):
        convergence_count += 1
    else:
        convergence_count = 0

# Use the trained perceptron to predict the class (virginica=1 or non-virginica=0) which is the output of this hidden
# layer neuron
for i in range(0, 150):
    hidden_virginica_outputs[i] = predict_class(
        inputs[i], hidden_virginica_weights)

# ========================== Output Layer: Setosa Artificial Neuron ==========================

# Weights and outputs of the output setosa neuron
output_setosa_weights = np.array([1, 0, -0.5])
output_setosa_outputs = np.zeros(150)

# Use the predefined weights to predict the class (setosa=1 or non-setosa=0) which is the output of this output layer
# neuron
for i in range(0, 150):
    x = np.array([hidden_setosa_outputs[i], hidden_virginica_outputs[i]])
    output_setosa_outputs[i] = predict_class(x, output_setosa_weights)

# ========================== Output Layer: Virginica Artificial Neuron ==========================

# Weights and outputs of the output virginica neuron
output_virginica_weights = np.array([0, 1, -0.5])
output_virginica_outputs = np.zeros(150)

# Use the predefined weights to predict the class (virginica=1 or non-virginica=0) which is the output of this output
# layer neuron
for i in range(0, 150):
    x = np.array([hidden_setosa_outputs[i], hidden_virginica_outputs[i]])
    output_virginica_outputs[i] = predict_class(x, output_virginica_weights)

# ========================== Output Layer: Versicolor Artificial Neuron ==========================

# Weights and outputs of the output versicolor neuron
output_versicolor_weights = np.array([-1, -1, 0.5])
output_versicolor_outputs = np.zeros(150)

# Use the predefined weights to predict the class (versicolor=1 or non-versicolor=0) which is the output of this output
# layer neuron
for i in range(0, 150):
    x = np.array([hidden_setosa_outputs[i], hidden_virginica_outputs[i]])
    output_versicolor_outputs[i] = predict_class(x, output_versicolor_weights)

calculate_accuracy(original_classes, output_setosa_outputs,
                   output_virginica_outputs, output_versicolor_outputs)

printClassification(original_classes, output_setosa_outputs,
                   output_virginica_outputs, output_versicolor_outputs)
