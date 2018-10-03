'''
This file is intended as a starting point
for the TF interface
'''
import nn_model
import tensorflow as tf
import tflowtools as tft
import numpy as np
import interface_helper as ih
import test_cases as tc
   
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

################################################
#  This is a scenario defining script as per   #
#  the assignment spec, if the user wants to   #
#  make a new neural net with new data/params  #
################################################
layers = [784,150,10] # Set up as a array where each index is the size of that layer.
hl_activation_function = 'relu'
ol_activation_function = 'softmax' #String: "softmax"
loss_function = "cross_entropy" #String: "MSE" or "cross_entropy"
learning_rate = 0.001
optimizer = "ADAM" #One of these: "GDS", "ADAM", "Adagrad", "RMS" as a string
mini_batch_size = 64
hm_times = 1000
weight_range = [-.1, .1] # Lower and upper bound
validation_interval = 5
#c_gen = (lambda: ih.load_data("/home/marius/ntnu/ai-progg/asgn1/data/wine.txt", 6, offset=3))
#c_gen = (lambda: ih.load_glass("/home/marius/ntnu/ai-progg/asgn1/data/glass.txt"))
c_gen = (lambda: ih.mnist_loader())
#x = nn_model.CaseManager(c_gen, vfrac=0.1, tfrac=0.1)
#y = x.get_training_cases()
#print(len(y))
vfrac = 0.1
tfrac = 0.1
case_frac = .2
map_bsize = 3
map_layers = [[0,"in"], [0, "out"]]
display_wb = False
wb_to_show = [[1,"wgt"], [1, "bias"]]


def create_network():
    """
    This method is the one high level function that functions as an interface
    where the user can define different inputs 
    """
    cman = nn_model.CaseManager(c_gen, case_frac=case_frac, vfrac=vfrac, tfrac=tfrac)
    with tf.Session() as sess:
        neural_net = nn_model.NeuralNetModel(layers, cman, weight_range, hl_activation_function, validation_interval, 
                                            mini_batch_size=mini_batch_size, learning_rate=learning_rate, sess = sess)
        neural_net.configure_training(loss_function, ol_activation_function,  optimizer)
        neural_net.run(steps=hm_times,bestk=1, map_bsize=map_bsize, map_layers=map_layers, show_wb=display_wb, wb=wb_to_show)
        


if __name__ == "__main__":
    #create_network()
    #tc.mnist_test(3, [[0, "in"], [1, "out"]], True, [[1, "bias"], [1, "wgt"]])
    #tc.mnist_test()
    #tc.vector_count_test()
    #tc.parity_test()
    #tc.symmetry_test()
    #tc.autoencoder_test()
    #tc.segment_counter_test()
    tc.wine_quality_test()
    
    
    input("Press [enter] to continue.")
    