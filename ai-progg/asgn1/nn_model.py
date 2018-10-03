import tensorflow as tf
import tflowtools as tft
import numpy as np
import os
import math
import mnist_basics as mn
import interface_helper as ih
import random
import matplotlib.pyplot as PLT

writer = tf.summary.FileWriter("myView")

class NeuralNetModel():
    '''
    This class represent a general neural net model
    '''
    def __init__(self, dims, cman, wr, hl_func, vint, mini_batch_size=32, learning_rate=0.01, sess=None):
        self.layer_sizes = dims # Size of each layer
        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate
        self.grabvars = [] # Variables to be monitored (FOR TENSORBOARD????)
        self.grabvars_figures = [] # One matplotlib figure for each grabvar
        self.layers = [] # Layers e.g modules 
        self.global_training_step = 0
        self.case_manager = cman #case manager easily handles dividing the dataset and shuffling it
        self.weight_range = wr # Initial weight range of the random weights. Used when building the neural net
        self.activation_function = hl_func # Hidden layer activation fucntion 
        self.validation_interval = vint # How often to run validation during trainign
        self.validation_history = [] # Error history of all validation runs
        self.sess = sess 
        self.create_model()

    def add_layer(self, layer): 
        self.layers.append(layer)

    def add_grabvar(self, layer_index, type='wgt'):
        self.grabvars.append(self.layers[layer_index].getvar(type))
        self.grabvars_figures.append(PLT.figure())

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
        # I add a value of outputlayer = true so that the hidden layer optimizer is not applied to the output 
        # layer. 
        for i, outsize in enumerate(self.layer_sizes[1:]):
            if i == (len(self.layer_sizes[1:]) - 1):
                layer = Layer(self, i, invar, insize, outsize, output_layer=True) 
            else:
                layer = Layer(self, i, invar, insize, outsize)
            invar = layer.output; insize = layer.outsize
        # The last layer is also the output layer/last layer of the whole neural network    
        self.output = layer.output        
        self.target = tf.placeholder(tf.float64, shape=(None, layer.outsize), name="Target")


    def configure_training(self, loss_function, ol_func, optimizer):
        self.predictor = self.output
        # Calculate loss 
        self.error = ih.gen_loss_function(loss_function, ol_func, self.predictor, self.target) 
        # Generate an optimizer 
        optimizer = ih.gen_optimizer(optimizer, self.learning_rate)
        self.trainer = optimizer.minimize(self.error)
        
    def do_training(self, sess, cases, hm_steps=100):
        self.error_history = []
        sess.run(tf.global_variables_initializer())
        gvars = self.error
        mbs = self.mini_batch_size; ncases = len(cases); nmb = math.ceil(hm_steps/mbs)
        # Would like to use epochs instead but assignment ask for steps
        # This runs a random minibatch of size N through the network 
        for step in range(hm_steps):
            error = 0 
            # Find a random integer between 0 and number of cases - mini batch size
            # so that we dont go out of bounce
            r = random.randint(0, (ncases - mbs))
            minibatch = cases[r:r+mbs]
            
            inputs = [c[0] for c in minibatch]; targets = [c[1] for c in minibatch]
            feeder = {self.input: inputs, self.target: targets}
            _,c = sess.run([self.trainer, self.error], feed_dict=feeder)
            #error += c            
            self.error_history.append((step, c))
            self.consider_validation(step, sess)
            if step % 100 is 0:
                print("Step #{} out of {} finnished. Loss {}".format(step, hm_steps, c))
            self.global_training_step += 1  

        #The prediction/output layer has never really applied softmax to it for later use
        # I use softmax when calculating error, but I never set it so I set it here so it is set for testing 
        self.predictor = tf.nn.softmax(self.predictor)
        ih.plot_training_history(self.error_history, validation_history=self.validation_history)
        
    def do_testing(self, sess, cases, msg="Testing", bestk=None, aa=False):
        inputs = [c[0] for c in cases]; targets = [c[1] for c in cases]
        feeder = {self.input: inputs, self.target: targets}
        self.test_func = self.error
        if bestk is not None and not aa:
            self.test_func = self.gen_match_counter(self.predictor, [tft.one_hot_to_int(list(v)) for v in targets], k = bestk)
        testres = sess.run([self.test_func], feed_dict=feeder)
        #self.run_one_step(self.test_func, self.grabvars, session=sess,
        #                            feed_dict=feeder)
        if bestk is None:
            h = 1
           # print('{} Set Error = {}'.format(msg, testres))
        else:
            print('%s Set Correct Classifications = %f %%' % (msg, 100*(testres[0]/len(cases))))
            # print("{} Set Correct Classifications = {}{}".format(msg, 100*(testres/len(cases))))
        return testres

    def do_mapping(self, sess, msg="Mapping", size=0, layers=None):
        c = self.case_manager.get_training_cases()
        c = c[:size] 
        for i in range(len(layers)): 
            self.add_grabvar(layers[i][0], layers[i][1])
        inputs = [n[0] for n in c]
        targets = [n[1] for n in c]
        feeder = {self.input: inputs, self.target: targets}
        res, grabs,_ = self.run_one_step(self.predictor, self.grabvars, session=sess, feed_dict=feeder)
        self.display_grabvars(grabs, self.grabvars,  step=self.global_training_step)

    def display_grabvars(self, grabbed_vals, grabbed_vars, step=1):
        name = [x.name for x in grabbed_vars]
        fig_index = 0
        for i, v in enumerate(grabbed_vals):
            if type(v) == np.ndarray and len(v.shape) > 1:
                tft.hinton_plot(v, fig=self.grabvars_figures[fig_index], 
                                title= name[i] + " at step " + str(step))
                fig_index += 1


    def gen_match_counter(self, logits, labels, k=1):
        correct = tf.nn.in_top_k(tf.cast(logits, tf.float32), labels, k)
        return tf.reduce_sum(tf.cast(correct, tf.int32))

    def training_session(self, epochs, sess=None, dir='test_dir'):
        session = sess if sess else tft.gen_initialized_session(dir=dir)
        self.current_session = session
        self.do_training(session, self.case_manager.get_training_cases(),epochs)
    
    def testing_session(self, sess, bestk=None):
        cases = self.case_manager.get_testing_cases()
        if len(cases) > 0:
            self.do_testing(sess, cases, msg="Final Testing", bestk=bestk)
        
    
    def test_on_trains(self,sess,bestk=None):
        self.do_testing(sess, self.case_manager.get_training_cases(), msg="Total Training", bestk=bestk)

    def consider_validation(self, epoch, sess):
        if self.validation_interval and (epoch % self.validation_interval == 0):
            cases = self.case_manager.get_validation_cases()
            if len(cases) > 0:
                error = self.do_testing(sess, cases)
                self.validation_history.append((epoch, error))

    def run_one_step(self, operators, grabbed_vars=None, dir='test_dir', session=None,
                    feed_dict=None, step=1):
        sess = session if session else tft.gen_initialized_session(dir=dir)
        results = sess.run([operators, grabbed_vars], feed_dict=feed_dict)
        return results[0], results[1], sess


    def run(self, steps=100, sess=None, bestk=None, map_bsize=0, map_layers=None):
        self.training_session(steps, sess=sess)
        self.test_on_trains(sess=self.current_session,bestk=bestk) 
        self.testing_session(sess=self.current_session,bestk=bestk)
        # Do map test if map batch size is > 0
        if map_bsize is not 0:
            self.do_mapping(self.current_session, size=map_bsize, layers=map_layers)
        tft.close_session(self.current_session, False)


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
        self.name = "Module-"+str(self.index)
        self.build()

    def getvar(self,type):
        return {"in": self.input, "out": self.output, "wgt": self.weights, "bias": self.biases}[type]

    def build(self):
        #self.weights = tf.Variable(tf.random_normal([self.insize, self.outsize],  seed=1), name="Weights")
        self.weights = tf.Variable(np.random.uniform(self.nn.weight_range[0], self.nn.weight_range[1], size=(self.insize,self.outsize)), name=self.name+"-weights")
        #self.biases = tf.Variable(tf.random_normal([self.outsize], seed=1), name="Bias")
        self.biases = tf.Variable(np.random.uniform(-.1, .1, size=(self.outsize)), name=self.name+"-biases")
        c = tf.add(tf.matmul(self.input, self.weights), self.biases)
        if not self.out:
            self.output = ih.gen_activation_function(self.nn.activation_function, c)
        else:
            self.output = c
        self.nn.add_layer(self)


class CaseManager():
    """
    A class for handling cases
    Taken from tutor3.py, but modified for my own usage
    This version also handles taking a fraction of size N of the whole dataset if it is specified. 
    """
    def __init__(self,cfunc, case_frac=1.0, vfrac=0,tfrac=0):
        self.casefunc = cfunc
        self.case_fraction = case_frac
        self.validation_fraction = vfrac
        self.test_traction = tfrac
        self.training_fraction = 1 - (vfrac + tfrac)
        self.generate_cases()
        self.organize_cases()

    def generate_cases(self):
        self.cases = self.casefunc()
    
    def organize_cases(self):
        c = np.array(self.cases)
        np.random.shuffle(c)
        cf = int(self.case_fraction * len(c))
        ca = c[:cf]
        print(len(ca))
        sep1 = round(len(ca) * self.training_fraction)
        sep2 = sep1 + round(len(ca) * self.validation_fraction)
        self.training_cases = ca[0:sep1]
        self.validation_cases = ca[sep1:sep2]
        self.testing_cases = ca[sep2:]
    
    def get_training_cases(self): return self.training_cases
    def get_validation_cases(self): return self.validation_cases
    def get_testing_cases(self): return self.testing_cases





