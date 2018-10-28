##################################################
# Implementation of games in this file. 
# The MCTS will be general and should work against
# any game implemented in this file
##################################################

# Need to parameters that specify each game of NIM:
# N = Number of pieces on the board
# K = Maximum number of pieces a player can take off the board during their turn,
# the minimum pieces to remove is ONE. 
# Given N & K, the rules of the game are: Each player take turn removing 
# pieces from the board and the last person to remove a piece is the winner. 


class GameState():

    WIN_STRING = "winner winner chicken dinner"    

    def __init__(self, num_pieces, max_pieces):
        self.num_pieces = num_pieces # Number of pieces on the board
        self.max_pieces = max_pieces # Maximum pieces a person is allowed to take per turn 


    def gen_successor(self):
        x = "TODO"
        


    def is_winner(self, move):
        if (self.num_pieces - move) == 0:
            return True


# Hva trenger jeg:
# Få vite hva slags state spillet er i 
# Generere barn for hver state
# Finne ut om man er i en leaf node/terminal node




    











