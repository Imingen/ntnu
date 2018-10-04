import interface_helper as ih
import nn_model as nn
import tensorflow as tf
import tflowtools as tft

def mnist_test(map_size=0, map_layers=None, display_wb=False, wb=None):
    layers = [784,150,10] # Set up as a array where each index is the size of that layer.
    hl_activation_function = 'relu'
    ol_activation_function = 'softmax' 
    loss_function = "cross_entropy" #String: "MSE" or "cross_entropy"
    learning_rate = 0.001
    optimizer = "ADAM" #"GDS", "ADAM", "Adagrad", "RMS" as a string
    mini_batch_size = 64
    hm_times = 1000
    weight_range = [-.1, .1] # Lower and upper bound
    validation_interval = 5
    c_gen = (lambda: ih.mnist_loader())
    vfrac = 0.1
    tfrac = 0.1
    case_frac = .2
    map_bsize = map_size
    map_layers = map_layers
    display_wb = display_wb
    wb_to_show = wb

    cman = nn.CaseManager(c_gen, case_frac=case_frac, vfrac=vfrac, tfrac=tfrac)
    with tf.Session() as sess:
        neural_net = nn.NeuralNetModel(layers, cman, weight_range, hl_activation_function, validation_interval, 
                                            mini_batch_size=mini_batch_size, learning_rate=learning_rate, sess = sess)
        neural_net.configure_training(loss_function, ol_activation_function,  optimizer)
        neural_net.run(steps=hm_times,bestk=1, map_bsize=map_bsize, map_layers=map_layers, show_wb=display_wb, wb=wb_to_show)

def vector_count_test(map_size=0, map_layers=None, display_wb=False, wb=None):
    layers = [15,30,15,16] # Set up as a array where each index is the size of that layer.
    hl_activation_function = 'tanh'
    ol_activation_function = 'softmax' 
    loss_function = "MSE" #String: "MSE" or "cross_entropy"
    learning_rate = 0.01
    optimizer = "ADAM" #"GDS", "ADAM", "Adagrad", "RMS" as a string
    mini_batch_size = 30
    hm_times = 900
    weight_range = [-.1, .1] # Lower and upper bound
    validation_interval = 10
    c_gen = (lambda: tft.gen_vector_count_cases(500,15))
    vfrac = 0.1
    tfrac = 0.1
    case_frac = 1.0
    map_bsize = map_size
    map_layers = map_layers
    display_wb = display_wb
    wb_to_show = wb

    cman = nn.CaseManager(c_gen, case_frac=case_frac, vfrac=vfrac, tfrac=tfrac)
    with tf.Session() as sess:
        neural_net = nn.NeuralNetModel(layers, cman, weight_range, hl_activation_function, validation_interval, 
                                            mini_batch_size=mini_batch_size, learning_rate=learning_rate, sess = sess)
        neural_net.configure_training(loss_function, ol_activation_function,  optimizer)
        neural_net.run(steps=hm_times,bestk=1, map_bsize=map_bsize, map_layers=map_layers, show_wb=display_wb, wb=wb_to_show)
               
def parity_test(map_size=0, map_layers=None, display_wb=False, wb=None):
    layers = [10,100,2] # Set up as a array where each index is the size of that layer.
    hl_activation_function = 'relu'
    ol_activation_function = 'softmax' 
    loss_function = "MSE" #String: "MSE" or "cross_entropy"
    learning_rate = 0.2
    optimizer = "GDS" #"GDS", "ADAM", "Adagrad", "RMS" as a string
    mini_batch_size = 100
    hm_times = 7000
    weight_range = [-.1, .1] # Lower and upper bound
    validation_interval = 10
    c_gen = (lambda: tft.gen_all_parity_cases(10))
    vfrac = 0.1
    tfrac = 0.1
    case_frac = 1.0
    map_bsize = map_size
    map_layers = map_layers
    display_wb = display_wb
    wb_to_show = wb

    cman = nn.CaseManager(c_gen, case_frac=case_frac, vfrac=vfrac, tfrac=tfrac)
    with tf.Session() as sess:
        neural_net = nn.NeuralNetModel(layers, cman, weight_range, hl_activation_function, validation_interval, 
                                            mini_batch_size=mini_batch_size, learning_rate=learning_rate, sess = sess)
        neural_net.configure_training(loss_function, ol_activation_function,  optimizer)
        neural_net.run(steps=hm_times,bestk=1, map_bsize=map_bsize, map_layers=map_layers, show_wb=display_wb, wb=wb_to_show)
            
def symmetry_test(map_size=0, map_layers=None, display_wb=False, wb=None):
    layers = [101,50,2] # Set up as a array where each index is the size of that layer.
    hl_activation_function = 'relu'
    ol_activation_function = 'softmax' 
    loss_function = "MSE" #String: "MSE" or "cross_entropy"
    learning_rate = 0.5
    optimizer = "GDS" #"GDS", "ADAM", "Adagrad", "RMS" as a string
    mini_batch_size = 40
    hm_times = 2500
    weight_range = [-.1, .1] # Lower and upper bound
    validation_interval = 10
    c_gen = (lambda: tft.gen_symvect_dataset(101, 2000))
    vfrac = 0.1
    tfrac = 0.1
    case_frac = 1.0
    map_bsize = map_size
    map_layers = map_layers
    display_wb = display_wb
    wb_to_show = wb

    cman = nn.CaseManager(c_gen, case_frac=case_frac, vfrac=vfrac, tfrac=tfrac)
    with tf.Session() as sess:
        neural_net = nn.NeuralNetModel(layers, cman, weight_range, hl_activation_function, validation_interval, 
                                            mini_batch_size=mini_batch_size, learning_rate=learning_rate, sess = sess)
        neural_net.configure_training(loss_function, ol_activation_function,  optimizer)
        neural_net.run(steps=hm_times,bestk=1, map_bsize=map_bsize, map_layers=map_layers, show_wb=display_wb, wb=wb_to_show)
              
def autoencoder_test(map_size=0, map_layers=None, display_wb=False, wb=None):
    layers = [8,2,8] # Set up as a array where each index is the size of that layer.
    hl_activation_function = 'sigmoid'
    ol_activation_function = None
    loss_function = "MSE" #String: "MSE" or "cross_entropy"
    learning_rate = 0.5
    optimizer = "GDS" #"GDS", "ADAM", "Adagrad", "RMS" as a string
    mini_batch_size = 2
    hm_times = 1000
    weight_range = [-.1, .1] # Lower and upper bound
    validation_interval = 10
    c_gen = (lambda: tft.gen_all_one_hot_cases(8))
    vfrac = 0.1
    tfrac = 0.1
    case_frac = 1.0
    map_bsize = map_size
    map_layers = map_layers
    display_wb = display_wb
    wb_to_show = wb


    cman = nn.CaseManager(c_gen, case_frac=case_frac, vfrac=vfrac, tfrac=tfrac)
    with tf.Session() as sess:
        neural_net = nn.NeuralNetModel(layers, cman, weight_range, hl_activation_function, validation_interval, 
                                            mini_batch_size=mini_batch_size, learning_rate=learning_rate, sess = sess)
        neural_net.configure_training(loss_function, ol_activation_function,  optimizer)
        neural_net.run(steps=hm_times,bestk=1, map_bsize=map_bsize, map_layers=map_layers, show_wb=display_wb, wb=wb_to_show)
              
def segment_counter_test(map_size=0, map_layers=None, display_wb=False, wb=None):
    layers = [25,65,50,20,9] # Set up as a array where each index is the size of that layer.
    hl_activation_function = 'relu'
    ol_activation_function = 'softmax' 
    loss_function = "MSE" #String: "MSE" or "cross_entropy"
    learning_rate = 0.01
    optimizer = "ADAM" #"GDS", "ADAM", "Adagrad", "RMS" as a string
    mini_batch_size = 128
    hm_times = 600
    weight_range = [-.1, .1] # Lower and upper bound
    validation_interval = 10
    c_gen = (lambda: tft.gen_segmented_vector_cases(25, 1000, 0, 8))
    vfrac = 0.1
    tfrac = 0.1
    case_frac = 1.0
    map_bsize = map_size
    map_layers = map_layers
    display_wb = display_wb
    wb_to_show = wb

    cman = nn.CaseManager(c_gen, case_frac=case_frac, vfrac=vfrac, tfrac=tfrac)
    with tf.Session() as sess:
        neural_net = nn.NeuralNetModel(layers, cman, weight_range, hl_activation_function, validation_interval, 
                                            mini_batch_size=mini_batch_size, learning_rate=learning_rate, sess = sess)
        neural_net.configure_training(loss_function, ol_activation_function,  optimizer)
        neural_net.run(steps=hm_times,bestk=1, map_bsize=map_bsize, map_layers=map_layers, show_wb=display_wb, wb=wb_to_show)
              
def wine_quality_test(map_size=0, map_layers=None, display_wb=False, wb=None):
    layers = [11,90,90,6] # Set up as a array where each index is the size of that layer.
    hl_activation_function = 'relu'
    ol_activation_function = 'softmax' 
    loss_function = "cross_entropy" #String: "MSE" or "cross_entropy"
    learning_rate = 0.01
    optimizer = "ADAM" #"GDS", "ADAM", "Adagrad", "RMS" as a string
    mini_batch_size = 128
    hm_times = 7500
    weight_range = [-.1, .1] # Lower and upper bound
    validation_interval = 1
    c_gen = (lambda: ih.load_data("/home/marius/ntnu/ai-progg/asgn1/data/wine.txt", 6, offset=3))
    x = nn.CaseManager(c_gen)
    vfrac = 0.1
    tfrac = 0.1
    case_frac = 1.0
    map_bsize = map_size
    map_layers = map_layers
    display_wb = display_wb
    wb_to_show = wb

    cman = nn.CaseManager(c_gen, case_frac=case_frac, vfrac=vfrac, tfrac=tfrac)
    with tf.Session() as sess:
        neural_net = nn.NeuralNetModel(layers, cman, weight_range, hl_activation_function, validation_interval, 
                                            mini_batch_size=mini_batch_size, learning_rate=learning_rate, sess = sess)
        neural_net.configure_training(loss_function, ol_activation_function,  optimizer)
        neural_net.run(steps=hm_times,bestk=1, map_bsize=map_bsize, map_layers=map_layers, show_wb=display_wb, wb=wb_to_show)
              
def yeast_test(map_size=0, map_layers=None, display_wb=False, wb=None):
    layers = [8,100,40, 30,10] # Set up as a array where each index is the size of that layer.
    hl_activation_function = 'relu'
    ol_activation_function = 'softmax' 
    loss_function = "MSE" #String: "MSE" or "cross_entropy"
    learning_rate = 0.01
    optimizer = "ADAM" #"GDS", "ADAM", "Adagrad", "RMS" as a string
    mini_batch_size = 32
    hm_times = 1000
    weight_range = [-.1, .1] # Lower and upper bound
    validation_interval = 1
    c_gen = (lambda: ih.load_data("/home/marius/ntnu/ai-progg/asgn1/data/yeast.txt",10, offset=1,delim=","))
    vfrac = 0.2
    tfrac = 0.1
    case_frac = 1.0
    map_bsize = map_size
    map_layers = map_layers
    display_wb = display_wb
    wb_to_show = wb

    cman = nn.CaseManager(c_gen, case_frac=case_frac, vfrac=vfrac, tfrac=tfrac)
    with tf.Session() as sess:
        neural_net = nn.NeuralNetModel(layers, cman, weight_range, hl_activation_function, validation_interval, 
                                            mini_batch_size=mini_batch_size, learning_rate=learning_rate, sess = sess)
        neural_net.configure_training(loss_function, ol_activation_function,  optimizer)
        neural_net.run(steps=hm_times,bestk=1, map_bsize=map_bsize, map_layers=map_layers, show_wb=display_wb, wb=wb_to_show)
  
def glass_test(map_size=0, map_layers=None, display_wb=False, wb=None):
    layers = [9,40,15,15,6] # Set up as a array where each index is the size of that layer.
    hl_activation_function = 'relu'
    ol_activation_function = 'softmax' 
    loss_function = "cross_entropy" #String: "MSE" or "cross_entropy"
    learning_rate = 0.001
    optimizer = "ADAM" #"GDS", "ADAM", "Adagrad", "RMS" as a string
    mini_batch_size = 128
    hm_times = 8000
    weight_range = [-.01, .01] # Lower and upper bound
    validation_interval = 10
    c_gen = (lambda: ih.load_glass("/home/marius/ntnu/ai-progg/asgn1/data/glass.txt"))
    vfrac = 0.1
    tfrac = 0.1
    case_frac = 1.0
    map_bsize = map_size
    map_layers = map_layers
    display_wb = display_wb
    wb_to_show = wb

    cman = nn.CaseManager(c_gen, case_frac=case_frac, vfrac=vfrac, tfrac=tfrac)
    with tf.Session() as sess:
        neural_net = nn.NeuralNetModel(layers, cman, weight_range, hl_activation_function, validation_interval, 
                                            mini_batch_size=mini_batch_size, learning_rate=learning_rate, sess = sess)
        neural_net.configure_training(loss_function, ol_activation_function,  optimizer)
        neural_net.run(steps=hm_times,bestk=1, map_bsize=map_bsize, map_layers=map_layers, show_wb=display_wb, wb=wb_to_show)
              
def fertility_test(map_size=0, map_layers=None, display_wb=False, wb=None):
    layers = [9,30,2] # Set up as a array where each index is the size of that layer.
    hl_activation_function = 'relu'
    ol_activation_function = 'softmax' 
    loss_function = "MSE" #String: "MSE" or "cross_entropy"
    learning_rate = 0.001
    optimizer = "ADAM" #"GDS", "ADAM", "Adagrad", "RMS" as a string
    mini_batch_size = 64
    hm_times = 300
    weight_range = [-.1, .1] # Lower and upper bound
    validation_interval = 10
    c_gen = (lambda: ih.load_fertility("/home/marius/ntnu/ai-progg/asgn1/data/fertility.txt"))
    vfrac = 0.1
    tfrac = 0.1
    case_frac = 1.0
    map_bsize = map_size
    map_layers = map_layers
    display_wb = display_wb
    wb_to_show = wb

    cman = nn.CaseManager(c_gen, case_frac=case_frac, vfrac=vfrac, tfrac=tfrac)
    with tf.Session() as sess:
        neural_net = nn.NeuralNetModel(layers, cman, weight_range, hl_activation_function, validation_interval, 
                                            mini_batch_size=mini_batch_size, learning_rate=learning_rate, sess = sess)
        neural_net.configure_training(loss_function, ol_activation_function,  optimizer)
        neural_net.run(steps=hm_times,bestk=1, map_bsize=map_bsize, map_layers=map_layers, show_wb=display_wb, wb=wb_to_show)
              
                   










