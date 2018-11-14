import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mplp  


class StateManager():

    def __init__(self, size):
        self.size = size
    
    def init_board(self):
        self.board = []
        for row in range(self.size):
            tmp = []
            for col in range(self.size):
                tmp.append(Piece(row, col, self.size))
            self.board.append(tmp)

    def print_board(self):
        for i, row in enumerate(self.board):
            c = []
            for j, col in enumerate(row):
                c.append(self.board[i][j].color)
            print(c)
    


    def check_state(self, player=1):
        if player == 1:
            current = None
            for piece in self.board[0]:
                if piece == 1:
                    current = piece
            








########################################################
#   FAGEDDABOUTID
########################################################

class Piece():

    def __init__(self, row, col, size, color='White'):
        self.row = row
        self.col = col
        self.color = color
        self.board_size = size
        self.init_relations()

    def init_relations(self):
        self.neighbors = []
        # Add neighbor UP
        if self.row - 1 >= 0:
            self.neighbors.append([self.row - 1, self.col])
        # Add neighbor DOWN
        if self.row + 1 <= self.board_size-1:
            self.neighbors.append([self.row + 1, self.col])
        # Add neighbor LEFT
        if self.col - 1 >= 0:
            self.neighbors.append([self.row, self.col - 1])
        # Add neighbor RIGHT
        if self.col + 1 <= self.board_size-1:
            self.neighbors.append([self.row, self.col + 1])
        # Add neighbor DIAGONAL UP
        if self.col + 1 <= self.board_size-1 and self.row - 1 >= 0:
            self.neighbors.append([self.row -1, self.col + 1])
        # Add neighbor DIAGONAL DOWN
        if self.col - 1 >= 0 and self.row + 1 <= self.board_size -1:
            self.neighbors.append([self.row + 1, self.col - 1])

    def __str__(self):
        return self.color





