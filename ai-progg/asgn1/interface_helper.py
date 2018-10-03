import tensorflow as tf
import matplotlib.pyplot as plt
import mnist_basics as mn
import numpy as np 

'''
This script file is intended to be used to generate TF operators (optimizers, loss functions etc)
based on the input from the user. 
It is mainly used to seperate the different functions so nn_model.py gets cleaned up
'''

def gen_optimizer(optimizer_name, learning_rate):
    if optimizer_name is "GDS":
        print("Optimizer: {}".format(optimizer_name))
        return tf.train.GradientDescentOptimizer(learning_rate, name="GDS")
    elif optimizer_name is "ADAM":
        print("Optimizer: {}".format(optimizer_name))
        return tf.train.AdamOptimizer(learning_rate, name="ADAM")
    elif optimizer_name is "Adagrad":
        print("Optimizer: {}".format(optimizer_name))
        return tf.train.AdagradOptimizer(learning_rate, name="Adagrad")
    elif optimizer_name is "RMS":
        print("Optimizer: {}".format(optimizer_name))
        return tf.train.RMSPropOptimizer(learning_rate, name="RMS")

def gen_loss_function(name, activation_function, prediction, target):
    if activation_function is "softmax":
        print("Output layer activation function: {}".format(activation_function))
        if name is "cross_entropy":
            print("Loss function: {}".format(name))
            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=target),  name="cross_entropy")
           # return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=target))
        if name is "MSE":
            print("Loss function: {}".format(name))
            prediction = tf.nn.softmax(prediction)
            return tf.reduce_mean(tf.square(tf.subtract(target, prediction)), name="MSE")

def gen_activation_function(name, layer):
    if name is "relu":
        return tf.nn.relu(layer, name="relu")
    if name is "tanh":
        return tf.nn.tanh(layer, name="tanh")
    if name is "sigmoid":
        return tf.nn.sigmoid(layer, name="sigmoid")

def plot_training_history(error_history, validation_history=[]):
    x_err = [n[0] for n in error_history]
    y_err = [n[1] for n in error_history]

    if len(validation_history) > 0:
        x_val = [n[0] for n in validation_history]
        y_val = [n[1] for n in validation_history]
        plt.plot(x_val, y_val,label="Validation history", zorder=3)

    plt.plot(x_err,y_err, linewidth=1, label="Error history", zorder=2)
    plt.title("Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend()
    plt.draw()
    plt.pause(0.001)

def mnist_loader(fscale=True):
    """
    function for loading the mnist dataset.
    This function loads the whole dataset, reshapes the input to correct form and onehots to labels. 
    If the user wants to only use parts of the dataset for training they need to specify "case_fraction"
    for the CaseManager class. 
    """
    images_raw, labels_raw = mn.load_mnist()

    images = []
    labels = []

    for image in images_raw:
        t = np.reshape(image, (784))
        images.append(t)

    for l in labels_raw:
        labels.append(one_hot_encoder(int(l), 10))
    
    if fscale:
        feature_scale_mnist(images)
    
    return [[im, lab] for im,lab in zip(images, labels)]

def load_data(filename, one_hot_size, delim=';', offset=0, fscale=True):
    """
    This will load data, mainly used for wine and yeast in this case. Glass got its own 
    loader since it was easier.
    Labels is one-hot-encoded and input/features are feature scaled unless specified otherwise.
    It returns the data as a case in the format that is accepted by the CaseManager. 
    Use offset to shift the on val in the vector e.x in wine where data starts at 3 offset would be 3
    to get a better label vector. 
    """
    data_file = open(filename, 'r')
    data = data_file.read().split('\n')
    lines = []
    labels = []
    for l in data:
        single_line = l.split(delim)
        sl_float = [float(x) for x in single_line]
        lines.append(sl_float[:-1])
        labels.append(sl_float[-1])

    labels2 = [] 
    for label in labels:
        label = int(label)
        x = one_hot_encoder(label, one_hot_size, offset=offset)
        labels2.append(x)
    if fscale:
        feature_scale(lines)
    return [[x, y] for x, y in zip(lines, labels2)]

def load_glass(filename, fscale=True):
    """
    This will load the glass.txt dataset. Created a specific function for glass-dataset
    since it needs a little work with the label one-hot vector. 
    """
    data_file = open(filename, 'r')
    data = data_file.read().split('\n')
    lines = []
    labels = []
    for i, l in enumerate(data):
        single_line = l.split(',')
        sl_float = [float(x) for x in single_line]
        lines.append(sl_float[:-1])
        labels.append(sl_float[-1])

    labels2 = [] 
    for label in labels:
        label = int(label)
        if label < 5:
            x = one_hot_encoder(label, 6, offset=1)
        if label > 4:
            x = one_hot_encoder(label, 6, offset=2)
        labels2.append(x)
    if fscale:
        feature_scale(lines)
    return [[x, y] for x, y in zip(lines, labels2)]

def feature_scale(d):
    """
    Feature scaling function as described in the assignment spec. 
    Input is the features to be scale i.e a matrix 
    """
    mean = np.mean(d, 0)
    var = np.std(d, 0)
    for i, elg in enumerate(d):
        for j, elem in enumerate(elg):
            d[i][j] =  (d[i][j] - mean[j]) / var[j]

def feature_scale_v2(d):
    """
    Another feature scale function as described in the assignment spec, 
    this one scales the features between 0 and 1.
    Input is the features to be scales i.e a matrix
    """
    max = np.max(d, axis=0)
    min = np.amin(d, axis=0)
    for i, elg in enumerate(d):
        for j, elem in enumerate(elg):
            d[i][j] = (d[i][j] - min[j]) / (max[j] - min[j])

def feature_scale_mnist(d):
    max = np.max(d, axis=1)
    min = np.amin(d, axis=1)
    for i, elg in enumerate(d):
        d[i] = (d[i] - min[i]) / (max[i] - min[i])


def one_hot_encoder(k, size, off_val=0.0, on_val=1.0, floats=False, offset=0):
    """
    A simple onehotencode-function. 
    Takes an int and the size of the vector and adds 0ff vals on all indexes
    except the kth element that gets the on-value. 
    This function also takes an offset if the one-hot vector needs to be shifted e.g
    the lowest element is not 0 and the user wants the lowest k to start at index 0. 
    """
    v = [off_val] * size
    v[k - offset] = on_val
    return v