import tensorflow as tf

'''
This script file is intended to be used to generate TF operators (optimizers, loss functions etc)
based on the input from the user. 
It is mainly used to seperate the different functions so nn_model.py gets cleaned up
'''

def gen_optimizer(optimizer_name, learning_rate):
    if optimizer_name is "GDS":
        print("Optimizer: {}".format(optimizer_name))
        return tf.train.GradientDescentOptimizer(learning_rate)
    elif optimizer_name is "ADAM":
        print("Optimizer: {}".format(optimizer_name))
        return tf.train.AdamOptimizer(learning_rate)
    elif optimizer_name is "Adagrad":
        print("Optimizer: {}".format(optimizer_name))
        return tf.train.AdagradOptimizer(learning_rate)
    elif optimizer_name is "RMS":
        print("Optimizer: {}".format(optimizer_name))
        return tf.train.RMSPropOptimizer(learning_rate, momentum=1.0)

def gen_loss_function(name, activation_function, prediction, target):
    if activation_function is "softmax":
        print("Output layer activation function: {}".format(activation_function))
        if name is "cross_entropy":
            print("Loss function: {}".format(name))
            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=target))
           # return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=target))
        if name is "MSE":
            print("Loss function: {}".format(name))
            prediction = tf.nn.softmax(prediction)
            #return tf.reduce_mean(tf.square(target - prediction))
            return tf.reduce_mean(tf.square(tf.subtract(target, prediction)))


