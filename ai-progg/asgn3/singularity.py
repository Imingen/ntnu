import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import numpy as np
import state_manager as sm



# region 
# def simulation(node, model):

#     n = node
#     while True:
        
#         flat_state = n.state.get_flat_board()
#         index = neural_magic(model, flat_state)
#         action = n.state.int_to_index(index)
#         print(index)
#         print(action)
#         new_state = n.state.gen_successor(action, n.player_num)
#         new_state.print_board()

#         if n.player_num == 1:
#             result = new_state.check_player1_win()
#             if result is True:
#                 # print('Player ONE')
#                 # n.state.print_board()
#                 n.state = new_state
#                 n.state.winner = "Player ONE"
#                 break
#         if n.player_num == 2:
#             result = new_state.check_player2_win()
#             if result is True:
#                 # print('Player TWO')
#                 # n.state.print_board()
#                 n.state = new_state
#                 n.state.winner = "Player TWO"
#                 break
#         new_node = Node(n, new_state, player_num=3-n.player_num)
#         n = new_node
#     n.state.print_board()
#     print(f"Winner: {n.state.winner}")

#     if n.player_num == 1:
#         return 1
#     else: 
#         return -1

# endregion

def get_anet(num_input, num_output):

    model = Sequential()
    model.add(Dense(num_input, activation='relu', input_shape=(num_input,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_output, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
              optimizer="SGD",
              metrics=['accuracy'])

    return model 

def neural_magic(model, state):
    """
    This lil bitch is for getting an action 
    """
    prediction = model.predict(state)
    # print(f"SUM:{sum(prediction[0])}")
    # print(f"Before: {prediction}")
    for i in range(len(prediction[0])):
        if state[0][i] != 0:
            prediction[0][i] = 0.0
    # print(f"After: {prediction}")
    # print(f"SUM:{sum(prediction[0])}")
    prediction = keras.utils.normalize(prediction[0], order=1)
    # print(f"SUM:{sum(prediction[0])}")
    #print(f"After: {prediction}")

    highest_index = np.argmax(prediction)
    return highest_index

# train = np.array([c1,c2])
# labels = np.array([l1,l2])

#model.summary()



# model.fit(x=train, y=labels, epochs=100)
# score = model.evaluate(x=train, y=labels)


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



