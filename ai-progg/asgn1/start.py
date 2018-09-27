'''
This file is intended as a starting point
for the TF interface
'''
import nn_model
import tensorflow as tf
import mnist_basics as mn
import numpy as np

def mnist_loader():
    images_raw, labels_raw = mn.load_mnist()

    images = []
    labels = []

    for image in images_raw:
        t = np.reshape(image, (784))
        images.append(t)

    labels = one_hot_encoder(labels_raw)
    
    return images, labels

def data_loader(filename):
    '''
    This will load the other data files 
    '''
    data_file = open(filename, 'r')
    data = data_file.read().split('\n')
    lines = []
    labels = []
    for l in data:
        single_line = l.split(';')
        lines.append(single_line[:-1])
        labels.append(single_line[-1])
    return lines, labels
   
def one_hot_encoder(labels):
    tmp = []
    for label in labels:
        one_hot = np.zeros(6)
        label_value = int(label[0])
        one_hot[label_value-3] = 1
        tmp.append(one_hot)
    return tmp


layers = [11, 700, 700, 700, 6] # Set up as a array where each index is the size of that layer.
hl_activation_function = 'relu'
ol_activation_function = 'softmax' #String: "softmax"
loss_function = "MSE" #String: "MSE" or "cross_entropy"
learning_rate = 0.001
#initial_weight_range = (0,0)
optimizer = "ADAM" #One of these: "GDS", "ADAM", "Adagrad", "RMS" as a string
input_data, input_labels = data_loader("/home/marius/ntnu/ai-progg/asgn1/data/wine.txt")
input_labels = one_hot_encoder(input_labels)
mini_batch_size = 8

def main_event():
    with tf.Session() as sess:
        neural_net = nn_model.NeuralNetModel(layers, mini_batch_size=mini_batch_size, learning_rate=learning_rate, sess = sess)
        neural_net.configure_training(loss_function, ol_activation_function, optimizer)
        neural_net.do_training(sess=sess, training_data=input_data, training_labels=input_labels)
        #create a model 
        #nn = nn_model()
        #train it
        #test it

if __name__ == "__main__":
    main_event()
    #data_path = "/home/marius/ntnu/ai-progg/asgn1/data/wine.txt"
    #data_loader(data_path)


