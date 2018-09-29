import tensorflow as tf
import tflowtools as tft
import numpy as np
import os
import math
import mnist_basics as mn
import interface_helper as ih
import random
import matplotlib.pyplot as PLT

writer = tf.summary.FileWriter("koko")

images_raw, labels_raw = mn.load_mnist()

images = []
labels = []

for image in images_raw:
    t = np.reshape(image, (784))
    images.append(t)

for label in labels_raw:
    one_hot = np.zeros(10)
    label_value = label[0]
    one_hot[label_value] = 1
    labels.append(one_hot)



class NeuralNetModel():
    '''
    This class represent a general neural net model
    '''
    def __init__(self, dims, mini_batch_size=32, learning_rate=0.01, sess=None):
        self.layer_sizes = dims # Size of each layer
        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate
        self.grabvars = [] # Variables to be monitored (FOR TENSORBOARD????)
        self.grabvars_figures = [] # One matplotlib figure for each grabvar
        self.layers = []
        self.sess = sess
        self.create_model()

    def add_layer(self, layer): 
        self.layers.append(layer)

    def add_grabvar(self, layer_index, type='wgt'):
        self.grabvars.append(self.layers[layer_index].getvar(type))
        self.grabvars_figures.appen(PLT.figure())

    def create_model(self):
        '''
        This method initializes the whole model with random weights and biases 
        The main point is to setup the different shapes for the matricses (n-shape arrays) that holds weights
        and vectors that holds the biases
        '''
        num_inputs = self.layer_sizes[0]
        # Placeholders usually x,y but more explainable like this. 
        self.input = tf.placeholder(tf.float64, shape=(None, num_inputs), name="Input")

        invar = self.input; insize = num_inputs
        # Build the structure of the neural net
        for i, outsize in enumerate(self.layer_sizes[1:]):
            if i == (len(self.layer_sizes[1:]) - 1):
                layer = Layer(self, i, invar, insize, outsize, output_layer=True) 
            else:
                layer = Layer(self, i, invar, insize, outsize)
            invar = layer.output; insize = layer.outsize
        self.output = layer.output
        self.target = tf.placeholder(tf.float64, shape=(None, layer.outsize), name="Target")
        #self.configure_training()

    def configure_training(self, loss_function, activation_function, optimizer):
        self.predictor = self.output # Simpel prediction runs will request the value of outpout neurons
        self.error = ih.gen_loss_function(loss_function, activation_function, self.predictor, self.target) 
        #self.error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.predictor, labels=self.target))
        # Defining the training operator
        #optimizer = tf.train.AdamOptimizer(self.learning_rate)
        optimizer = ih.gen_optimizer(optimizer, self.learning_rate)
        self.trainer = optimizer.minimize(self.error)
        
    def do_training(self, sess, cases, hm_epochs=100):
        self.error_history = []
        sess.run(tf.global_variables_initializer())
        print("Prediction: {}".format(self.predictor))
        ok = int(len(training_data)*0.8)
        ok2 = int(len(training_data)*0.)
        t_d = training_data[:ok]
        t_l = training_labels[:ok]
        filename = "epoch"
        for epoch in range(epochs):
            epoch_error = 0 
            gvars = [self.error] + self.grabvars
            mbs = self.mini_batch_size; ncases = len(cases); nmb = math.ceil(ncases/mbs)
            while i < len(t_d):
                mini_batch_loss = 0
                mbs_num += 1
                start = i
                end = i + mbs
                epoch_x = t_d[start:end]
                #if epoch == 1 or epoch == 9:
                #    if i == 1 or i == 9:
                #        f = open(filename_n, "a")
                #        f.write(str(epoch_x[0][0]))
                #if epoch == 9:
                #    f = open("epoch9.txt", "a")
                #    f.write(str(epoch_x[0][0]))
                epoch_y = t_l[start:end]

                _, c = sess.run([self.trainer, self.error], 
                                feed_dict={self.input: epoch_x, self.target: epoch_y})
                mini_batch_loss += c
                epoch_error += c
                i += mbs
               # print("Minibatch #{} completed out of {}. Loss: {}".format(mbs_num, math.ceil((len(training_data)/mbs)), mini_batch_loss))
                #print('Minibatch #', mbs_num, "completed out . Loss:", mini_batch_loss)
            if epoch % 1000 is 0: 
                print('Epoch', epoch + 1, 'completed out of', epochs, 'loss:', epoch_error)
            
        #correct = tf.equal(tf.argmax(self.predictor, 1), tf.argmax(self.target, 1))
        correct = tf.nn.in_top_k(tf.cast(self.predictor, tf.float32), [tft.one_hot_to_int(list(v)) for v in self.target] , k=1)
        accuracy = tf.reduce_mean(tf.cast(correct,'float'), name="XD")
        print('Accuracy:', accuracy.eval({self.input:training_data[-ok2:], self.target:training_labels[-ok2:]}))    

        writer.add_graph(sess.graph)

        #self.do_testing(self.sess, training_data, training_labels)

    def do_testing(self, sess, input_data, labels, bestk=None):
        tmp = zip(input_data, labels)
        random.shuffle(tmp)
        inp, lab = zip(*tmp)
                        
#region Sentdex style
        # Filtrerer ut de forskjellige layerene: input, hidden og output
        #input_nodes = self.nodes[1]
        #output_nodes = self.nodes[-1]
        #hidden_nodes = self.nodes[1:-1]

        # Først lagrer vi random weights og biases for hvert layer
        # Her spesifiserer vi også shape på hvert layer siden dette kan være forskjellig for hvert layer
        #layer_setup = []
        #for number in range(0, self.num_layers-1):
            # Weights = Alle weights mellom hver node i layer n og alle noder i layer n+1
            # Biases = 1-dimensjonal vektor
        #    layer = {"weights": tf.Variable(tf.random_normal([self.nodes[number], self.nodes[number+1]])),
        #            "biases": tf.Variable(tf.random_normal([self.nodes[number+1]]))}
        #    layer_setup.append(layer)

        # Outputlayer må appendes til slutt fordi: 
        # Siste elementet kan ikke knytters sammeen med number+1 e.g indexoutofbounds
        #out_put_layer = {"weights": tf.Variable(tf.random_normal([self.nodes[-2], self.nodes[-1]])),
        #               "biases": tf.Variable(tf.random_normal([self.nodes[-1]]))}
        #layer_setup.append(out_put_layer)

        # Alle layers er satt opp
        #
        #layer1 = tf.add(tf.matmul(lay))
#endregion 

class Layer():
    '''
    One layer in the nerual net. 
    Holds incoming weights and biases + the neurons e.g the output of this layer
    '''
    def __init__(self,nn,index,input_var, insize, num_neur, output_layer=False):
        self.nn = nn # a pointer to the nerual net model this layer is a part of
        self.index = index
        self.input = input_var # Either the input of the neural net or the output of the previous layer
        self.insize = insize # Number of neurons feeding into this layer
        self.outsize = num_neur # Number of neurons in this layer
        self.out = output_layer
        self.build()

    def build(self):
        #self.weights = tf.Variable(tf.random_normal([self.insize, self.outsize]), name="Weights")
        self.weights = tf.Variable(np.random.uniform(-.1, .1, size=(self.insize,self.outsize)), name="Weights")
        #self.biases = tf.Variable(tf.random_normal([self.outsize]), name="Bias")
        self.biases = tf.Variable(np.random.uniform(-.1, .1, size=(self.outsize)), name="Biases")
        c = tf.add(tf.matmul(self.input, self.weights), self.biases)
        if not self.out:
            self.output = tf.nn.relu(c)
        else:
            self.output = tf.nn.relu(c)
        self.nn.add_layer(self)


if __name__ == "__main__":
    with tf.Session() as sess:
        nn = NeuralNetModel([784, 200, 200, 200, 10], 128, learning_rate=0.1, sess=sess)
        nn.do_training(sess)
        writer.add_graph(sess.graph)



