import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU
from keras import optimizers
from keras import regularizers
import numpy as np
import state_manager as sm
import random 



class NeuralNet():

    def __init__(self, lr, num_input, layers, split=0.8, optimizer='adam',
            loss='mean_squared_error', epochs=10, activation='relu', output_activation='softmax'):
        self.learning_rate = lr
        self.layers = layers
        self.num_input = num_input
        self.split = 0.8
        self.optimizer = optimizer
        self.loss = loss
        self.activation = activation
        self.out_activation = output_activation
        self.epochs = epochs


    def get_anet(self):

        model = Sequential()

        model.add(Dense(self.num_input+1,  input_shape=((self.num_input)+1,)))

        for i in range(len(self.layers)):
            model.add(Dense(self.layers[i], activation=self.activation))
            model.add(Dropout(0.3))

        model.add(Dense((self.num_input), activation=self.out_activation))
        if self.optimizer == 'adam':
            opt = optimizers.Adam(lr=self.learning_rate)
        if self.optimizer == 'sdg':
            opt = optimizers.SGD(lr=self.learning_rate)
        if self.optimizer == 'rmsprop':
            opt = optimizers.RMSprop(lr=self.learning_rate)
        if self.optimizer == 'adagrad':
            opt = optimizers.Adagrad(lr=self.learning_rate)

        model.compile(loss=self.loss,
                optimizer=opt,
                metrics=['accuracy'])

        return model 

    def neural_magic(self, model, state, player, epsilon=0.1):
        """
        This lil bitch is for getting an action 
        """
        highest_index = 0

        flat_state = state.get_flat_board(player)
        prediction = model.predict(flat_state)
        tmp = flat_state[0][1:]
        for i in range(len(prediction[0])):
            if tmp[i] != 0:
                prediction[0][i] = 0.0

        prediction = keras.utils.normalize(prediction[0], order=1)

        if epsilon is not None:
            epsilon = epsilon
            check = random.randint(0, 100)
            check = check/100
            if check > epsilon:
                tmp = np.argmax(prediction)
                highest_index = state.int_to_index(tmp)
            else:
                lac = state.get_legal_actions()
                try:
                    r = random.randint(0,  len(lac)-1)
                except:
                    print(state.winner)
                    print(flat_state)
                    state.print_board_pretty()
                    print(f"LAC: {lac}")
                return lac[r]
        else:
            tmp = np.argmax(prediction)
            highest_index =  state.int_to_index(tmp)

        return highest_index

    def train_anet(self, model, train_data):

        x = np.array([x[0] for x in train_data])
        y = np.array([x[1] for x in train_data])
        y = keras.utils.normalize(y, order=1)
        
        split = int((len(x) * self.split))
        train = x[:split]
        train_labels = y[:split]

        test = x[split:]
        test_labels = y[split:]

        model.fit(x=train, y=train_labels, validation_split=0.1, epochs=self.epochs, verbose=1)
        score = model.evaluate(x=test, y=test_labels)
        print(score)


