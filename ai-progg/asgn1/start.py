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
    feature_scale(lines)
    return [[x, y] for x, y in zip(lines, labels2)]
   

def data_loader2(filename):
    '''
    This will load the other data files 
    '''
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
    feature_scale(lines)
    return [[x, y] for x, y in zip(lines, labels2)]

def one_hot_encoder(k, size, off_val=0.0, on_val=1.0, floats=False, offset=0):
    v = [off_val] * size
    v[k - offset] = on_val
    return v

def gen_many_onehot_cases(hm):
    for i in range(hm):
        return [[c,c] for c in tft.all_one_hots(8)]

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

#
layers = [9,20,40,50,100,50,40,20,6] # Set up as a array where each index is the size of that layer.
hl_activation_function = 'relu'
ol_activation_function = 'softmax' #String: "softmax"
loss_function = "cross_entropy" #String: "MSE" or "cross_entropy"
learning_rate = 0.001
optimizer = "ADAM" #One of these: "GDS", "ADAM", "Adagrad", "RMS" as a string
mini_batch_size = 1
hm_times = 71
weight_range = [-.1, .1] # Lower and upper bound
validation_interval = 5

#c_gen = (lambda: tft.gen_segmented_vector_cases(25, 1000, 0, 8))
c_gen = (lambda: data_loader2("/home/marius/ntnu/ai-progg/asgn1/data/glass.txt"))
#c_gen = (lambda: tft.gen_all_bit_vectors(4))
cman = nn_model.CaseManager(c_gen, vfrac=0.15, tfrac=0.15)
#x = cman.get_training_cases()
#print(x[:3], "\n")
#for i in x:
#  print(i)
#  print(len(x))


def main_event():
    with tf.Session() as sess:
        neural_net = nn_model.NeuralNetModel(layers, cman, weight_range, hl_activation_function, validation_interval, mini_batch_size=mini_batch_size, learning_rate=learning_rate, sess = sess)
        neural_net.configure_training(loss_function, ol_activation_function,  optimizer)
        neural_net.run(steps=hm_times,bestk=1)
        
if __name__ == "__main__":
    main_event()
    input("Press [enter] to continue.")
    