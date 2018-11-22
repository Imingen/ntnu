import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
import numpy as np
import state_manager as sm
import random 



def get_anet(num_input, num_output):

    model = Sequential()
    model.add(Dense(num_input+1, activation='relu', input_shape=(num_input+1,)))
    model.add(Dense(254, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_output, activation='softmax'))
    adam = optimizers.Adam(lr=0.001)
    model.compile(loss='mean_squared_error',
              optimizer=adam,
              metrics=['accuracy'])

    return model 

def neural_magic(model, state, epsilon=True):
    """
    This lil bitch is for getting an action 
    """
    if epsilon:
        epsilon = 0.1
        check = random.randint(0, 100)
        check = check/100

    prediction = model.predict(state)
    # print(f"SUM:{sum(prediction[0])}")
    # print(f"Before: {prediction}")
    tmp = state[0][1:]
    for i in range(len(prediction[0])):
        if tmp[i] != 0:
            prediction[0][i] = 0.0
    # print(f"After: {prediction}")
    # print(f"SUM:{sum(prediction[0])}")
    prediction = keras.utils.normalize(prediction[0], order=1)
    # print(f"SUM:{sum(prediction[0])}")
    #print(f"After: {prediction}")
    if epsilon:
        if check > epsilon:
            highest_index = np.argmax(prediction)
        else:
            r = random.randint(0, len(prediction[0])-1)
            return r
    else:
        return np.argmax(prediction)
    return highest_index

def train_anet(model, train_data):

    #print(train_data)
    train = np.array([x[0] for x in train_data])
    labels = np.array([x[1] for x in train_data])
    labels = keras.utils.normalize(labels, order=1)
    #print(labels)
    # model.summary()
    model.fit(x=train, y=labels, epochs=10, verbose=0)
    score = model.evaluate(x=train, y=labels)
    #print(score)


#if __name__ == "__main__":
    
    # state = sm.StateManager(4)
    # state.init_board()
    # root = Node(None, state)

    # flat_state = state.get_flat_board()
    # #print(flat_state)
    # anet = get_anet(state.size**2, state.size**2)
    # #neural_magic(anet, flat_state)
    # simulation(root, anet)


#print("_------------------_")
#print(the_board)
#ok = model.predict_classes(the_board)
#print(ok)



