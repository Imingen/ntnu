import numpy as np 
import random 

######################################
#
# 1. Initialization of weights with random values
# 2. Treshold Sigma should also be a random value in initialisation
# 3. Activate the perceptron with inputs and desired outputs
# 4. Update the weights 
# 5. Iterate. Go back to step 3.
#
# Learning is done by making small adjustment to weights to minimize
# the difference between observed and predicted values
# 
######################################



class perceptron():

    def __init__(self, lrate):
        self.learning_rate = lrate
        self.treshold = random.uniform(-0.5, 0.5)
        self.weights = np.random.uniform(low=(-0.5), high=0.5, size=(2))

    
    def activation(self, inputs):
        self.inputs = inputs
        weighted_sum = 0
        for i, w in enumerate(self.weights):
            zum = self.inputs[i] * self.weights[i] - self.treshold
            weighted_sum += zum
        
        perceptron_output = 1 if weighted_sum > 0 else 0
        return perceptron_output    
    
    def update_weights(self, prediction, label):
        self.error = label - prediction
        for i, w in enumerate(self.weights):
            delta_rule = self.learning_rate * self.error * self.inputs[i]
            #print("1st", self.weights[i])
            #print("delta rule: ", delta_rule)
            self.weights[i] += delta_rule
            #print("2nd", self.weights[i])


        

inputs = [(0,0), (0,1), (1,0), (1,1)]
and_labels = [0, 0, 0, 1]
or_labels = [0, 1, 1, 1]
epochs = 100

perceptron = perceptron(1)

for epoch in range(epochs):
    print(f"Epcoh #{epoch+1}")
    print("######################################################")
    for i, inp in enumerate(inputs):
        prediction = perceptron.activation(inputs[i])
        print(f"Input: {inputs[i]}. Desired output: {and_labels[i]}. Prediction: {prediction}. \nInitial weights: {perceptron.weights}.")
        #print(f"Weights @ epoch #{epoch+1} & iteration #{i+1}: {perceptron.weights}")
        perceptron.update_weights(prediction, and_labels[i])
        
        print(f"Error: {perceptron.error}.\nUpdated weights: {perceptron.weights} \n")
    print("######################################################")
    
    





