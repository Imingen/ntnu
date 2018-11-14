import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mplp  
import copy


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
    
    def get_legal_actions(self):
        legal_actions = []
        for i, row in enumerate(self.board):
            for j, col in enumerate(row):
                if self.board[i][j].color is 'White':
                    legal_actions.append([i, j])
        return legal_actions

    def do_move(self, piece):
        i = piece.row
        j = piece.col
        self.board[i][j] = piece

    def check_piece(self, piece, board, player):
        if player == 1 and piece.row == self.size-1:
            return True
        if player == 2 and piece.col == self.size-1:
            return True
        c = [x for x in piece.neighbors]
        for neighbor in c:
            if board[neighbor[0]][neighbor[1]].color == "R":
                board[piece.row][piece.col].color = "CHECKED"
                return self.check_piece(board[neighbor[0]][neighbor[1]], board, player)
            else:
                continue

    def check_state(self, player=1):
        """This code is garbage
        TODO: Refractor lil bitch
        """
        cp = copy.deepcopy(self.board)
        if player == 1:
            for piece in self.board[0]:
                if piece.color == "R":
                    res = self.check_piece(piece, cp, player)
                    if res == True:
                        return res
                    else:
                        continue
            return False
        elif player == 2:
            for piece in self.board:
                if piece[0].color == "R":
                    res = self.check_piece(piece[0], cp, player)
                    if res == True:
                        return res
                    else:
                        continue
            return False


            
class Piece():
    """Represents ONE piece on the HEX board.
    """

    def __init__(self, row, col, size, color='W'):
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








#######################################################
#   Code Graveyard
#######################################################

# def check_state(self, piece):
#     for piece in self.board[0]:
#         board_copy = copy.deepcopy(self.board)
#         if piece.color == "Red":
#             current = piece
#             c = [x for x in piece.neighbors]
#             if any([x for x in c if self.board[x[0]][x[1]].color == 'Red']):
#                 print(c)
#                 print("HEKK")



