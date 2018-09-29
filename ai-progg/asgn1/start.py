'''
This file is intended as a starting point
for the TF interface
'''
import nn_model
import tensorflow as tf
import tflowtools as tft
import mnist_basics as mn
import numpy as np

def mnist_loader():
    images_raw, labels_raw = mn.load_mnist()

    images = []
    labels = []

    for image in images_raw:
        t = np.reshape(image, (784))
        images.append(t)

    for l in labels_raw:
        labels.append(one_hot_encoder(int(l), 10))
    
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
        sl_float = [float(x) for x in single_line]
        lines.append(sl_float[:-1])
        labels.append(sl_float[-1])

    
    labels2 = [] 
    for label in labels:
        label = int(label)
        x = one_hot_encoder(label, 6, offset=3)
        labels2.append(x)
    return lines, labels2
   
def one_hot_encoder(k, size, off_val=0.0, on_val=1.0, floats=False, offset=0):
    v = [off_val] * size
    v[k - offset] = on_val
    return v

def feature_scale(d):
    mean = np.mean(d, 0)
    var = np.std(d, 0)
    for i, elg in enumerate(d):
        for j, elem in enumerate(elg):
            d[i][j] =  (d[i][j] - mean[j]) / var[j]

def feature_scale_v2(d):
    max = np.max(d, axis=0)
    min = np.amin(d, axis=0)
    for i, elg in enumerate(d):
        for j, elem in enumerate(elg):
            d[i][j] = (d[i][j] - min[j]) / (max[j] - min[j])

def parity():
    x = tft.gen_all_parity_cases(16)
    new_x = []
    input_data = []
    labels = []
    for i, elem in enumerate(x):
        new_x.append(elem)            

    for i, x in enumerate(new_x):
        input_data.append(new_x[i][0])
        labels.append(new_x[i][1])
    print(input_data)
    return input_data, labels

x = tft.gen_vector_count_cases(500,10)
input_data = [c[0] for c in x]
input_labels = [c[1] for c in x]
#print(input_data[:1])
#print(input_labels[:1])


layers = [10, 30, 11] # Set up as a array where each index is the size of that layer.
hl_activation_function = 'relu'
ol_activation_function = 'softmax' #String: "softmax"
loss_function = "MSE" #String: "MSE" or "cross_entropy"
learning_rate = 0.5
#initial_weight_range = (0,0)
optimizer = "GDS" #One of these: "GDS", "ADAM", "Adagrad", "RMS" as a string
#input_data, input_labels = mnist_loader() #data_loader("/home/marius/ntnu/ai-progg/asgn1/data/wine.txt")
#input_data, input_labels = parity()
mini_batch_size = 20

#feature_scale(input_data)

def main_event():
    with tf.Session() as sess:
        neural_net = nn_model.NeuralNetModel(layers, mini_batch_size=mini_batch_size, learning_rate=learning_rate, sess = sess)
        neural_net.configure_training(loss_function, ol_activation_function, optimizer)
        #neural_net.do_training(sess=sess, training_data=input_data, training_labels=input_labels)
        neural_net.do_training(sess=sess, cases)

        

if __name__ == "__main__":
    main_event()
    #parity()
  #  data_path = "/home/marius/ntnu/ai-progg/asgn1/data/wine.txt"
    #data_loader(data_path)

