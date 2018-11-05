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


    def __init__(self, num_pieces, max_pieces):
        self.num_pieces = num_pieces # Number of pieces on the board
        self.max_pieces = max_pieces # Maximum pieces a person is allowed to take per turn 
        #self.player_one = 0
        #self.player_two = 0
        #self.players_turn = None


    def gen_successor(self, action):
        '''
        Generate a successor state from this state based on the action taken,
        where an action is an integer representing the number of pieces taken from the board. 
        '''
        num_sticks = self.num_pieces - action # The number of sticks in the next state is the number of sticks in this state minus the action
        new_state = GameState(num_sticks, self.max_pieces)
        return new_state


    def get_legal_actions(self):
        '''
        Get all the possible actions from this state
        '''
        actions = []
        if self.num_pieces != 0:
            if self.num_pieces >= self.max_pieces:
                for i in range(1, self.max_pieces+1):
                    actions.append(i)
            else:
                for i in range(1, self.num_pieces+1):
                    actions.append(i)
            # for i in actions:
            #     print(f"action: {i}")
            return actions


    def do_move(self, move):
        if move > 0 and move <= self.max_pieces:
            self.num_pieces -= move
    
    def is_winner(self):
        return True if self.num_pieces == 0 else False

      #  if self.num_pieces == 0:
     #       return True
     #   else:
      #      return False
