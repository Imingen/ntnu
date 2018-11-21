import keras
from state_manager import StateManager
from singularity import neural_magic
import numpy as np

model_1 = keras.models.load_model("/home/marius/ntnu/ai-progg/asgn3/models/anet1.h5")
#model_2 = keras.models.load_model("/home/marius/ntnu/ai-progg/asgn3/models/anet20.h5")
#model_3 = keras.models.load_model("/home/marius/ntnu/ai-progg/asgn3/models/anet40.h5")
#model_4 = keras.models.load_model("/home/marius/ntnu/ai-progg/asgn3/models/anet60.h5")
model_5 = keras.models.load_model("/home/marius/ntnu/ai-progg/asgn3/models/anet6.h5")
# flat_state = n.state.get_flat_board()
# index = neural_magic(model, flat_state)
# action = n.state.int_to_index(index)
# print(index)
# print(action)
# new_state = n.state.gen_successor(action, n.player_num)

state = StateManager(4)
state.init_board()
player_1 = 1
player_2 = 2
model_dict = {1: model_5, 2: model_1}
current = player_1

while not state.is_winner():

    flat_state = state.get_flat_board()
    flat_state = np.(flat_state, 0, current)
    index = neural_magic(model_dict[current], flat_state, epsilon=False)
    action = state.int_to_index(index)
    state.do_move(action, current)
    state.print_board_pretty()
    current = 3 - current
    if state.is_winner():
        break

print(state.winner)









