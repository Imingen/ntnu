import keras
from state_manager import StateManager
from singularity import neural_magic
import numpy as np

model_0 = keras.models.load_model("/home/marius/ntnu/ai-progg/asgn3/models/0.h5")
model_5 = keras.models.load_model("/home/marius/ntnu/ai-progg/asgn3/models/5.h5")
model_10 = keras.models.load_model("/home/marius/ntnu/ai-progg/asgn3/models/10.h5")
model_15 = keras.models.load_model("/home/marius/ntnu/ai-progg/asgn3/models/15.h5")
model_20 = keras.models.load_model("/home/marius/ntnu/ai-progg/asgn3/models/20.h5")
model_25 = keras.models.load_model("/home/marius/ntnu/ai-progg/asgn3/models/25.h5")
model_30 = keras.models.load_model("/home/marius/ntnu/ai-progg/asgn3/models/30.h5")



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
model_dict = {1:model_5 , 2: model_30}
current = player_2

while state.is_winner() == False:
    print(f"Player doing move: {current}")
    flat_state = state.get_flat_board(current)
   # flat_state = np.insert(flat_state, 0, current)
   # flat_state = np.array([flat_state])
    print(flat_state)
    index = neural_magic(model_dict[current], flat_state, epsilon=False)
    action = state.int_to_index(index)
    state.do_move(action, current)
    state.print_board_pretty()
    if state.is_winner():
        print("HEHEHHEHEHHEHE")
        print(state.winner)

        break
    current = 3 - current


print(state.winner)









