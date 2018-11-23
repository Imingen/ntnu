import keras
from state_manager import StateManager
from singularity import NeuralNet
import numpy as np
import pickle
import random
import os


class TOPP():

    def __init__(self, net_man, num_games, board_size, path):
        self.num_games = num_games
        self.board_size = board_size
        self.path = path
        self.net_man = net_man
        self.tournament_list = []
        self.win_stats = []
        self.load_models()
    
    def load_models(self):
        directory = os.fsencode(self.path)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".h5"): 
                model = keras.models.load_model(self.path + filename)
                self.tournament_list.append([filename, model])
                self.win_stats.append([filename, 0])
        self.num_models = len(self.tournament_list)
       # print(self.tournament_list)
       # print(self.win_stats)


    def run_tournament(self, verbose=False):
        t = self.num_models
        for i in range(t):
            for j in range(i+1, t):
                if verbose:
                    print(f"Model{i+1} vs Model{j+1}")
                for game in range(1, self.num_games+1):
                    if verbose:
                        print(f"Starting game #{game}")
                    winner = self.model_vs_model(self.tournament_list[i][1], self.tournament_list[j][1], verbose = verbose)
                    if winner == 1:
                        self.win_stats[i][1] += 1
                    else:
                        self.win_stats[j][1] += 1
        self.win_stats = sorted(self.win_stats,key=lambda x: x[1])
        print(self.win_stats)


    def model_vs_model(self, model_1, model_2, verbose=False):
        state = StateManager(self.board_size)
        state.init_board()
        player_1 = 1
        player_2 = 2

        model_dict = {1: model_1, 2: model_2}
        current = random.randint(1,2)
        if verbose:
            print(f"Player {current} is starting")

        while state.is_winner() == False:
            action = self.net_man.neural_magic(model_dict[current], state, current,  epsilon=None)
            state.do_move(action, current)
            #state.print_board_pretty()
            if state.is_winner():
                break
            current = 3 - current
        if verbose:
            print(state.winner)
        return current




if __name__ == "__main__":
    topp = TOPP(1, 3, path="/home/marius/ntnu/ai-progg/asgn3/models/")
    #topp.model_vs_model(model_0, model_20)
    topp.run_tournament(verbose=True)
#    topp.load_models()









