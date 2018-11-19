import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mplp  
import copy


class StateManager():


    def __init__(self, size):
        self.size = size
        self.color_dict = {1:"R", 2:"B", 3:"W"}
        self.winner = None
        self.checked_nodes = []
    
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
        for i in range(0, self.size):
            for j in range(0, self.size):
                if self.board[i][j].color is "W":
                    legal_actions.append([i, j])
        return legal_actions

    def gen_successor(self, action, player):
        new_state = StateManager(self.size)
        new_state.board = copy.deepcopy(self.board)
        new_state.do_move(action, player)
        return new_state

    def do_move(self, square, player):
        i = square[0]
        j = square[1]
        if self.board[i][j].color == self.color_dict[3]:
            self.board[i][j] = Piece(i, j, self.size, self.color_dict[player])


    def check_piece(self, piece, board, player):
        """Recursion is yolo m8
        """
        result = False
        if player == 1 and piece.row == self.size-1:
            return True
        if player == 2 and piece.col == self.size-1:
            return True

        self.checked_nodes.append(piece)

        for node in self.checked_nodes:
            c = [x for x in piece.neighbors]
            for neighbor in c:
                current = board[neighbor[0]][neighbor[1]]
                if current in self.checked_nodes:
                    continue
                elif board[neighbor[0]][neighbor[1]].color == self.color_dict[player]:
                    result = self.check_piece(current, board, player)
                    if result == True:
                        return result
        return result

    def check_player1_win(self):
        result = False
        for piece in self.board[0]:
            if piece.color == self.color_dict[1]:
                self.checked_nodes = []
                result = self.check_piece(piece, self.board, player = 1)
                if result is True:
                    break
                else:
                    continue
        return result

    def check_player2_win(self):
        result = False
        for piece in self.board:
            if piece[0].color == self.color_dict[2]:
                self.checked_nodes = []
                result = self.check_piece(piece[0], self.board, player = 2)
                if result is True:
                    break
                else:
                    continue
        return result

            
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

# def check_piece_v2(self, piece, board, player):
#     if player == 1 and piece.row == self.size-1:
#         self.winner = 'one'
#         return True
#     if player == 2 and piece.col == self.size-1:
#         self.winner = 'two'
#         return True
#     c = [x for x in piece.neighbors]
#     for neighbor in c:
#         if board[neighbor[0]][neighbor[1]].color == self.color_dict[player]:
#             board[piece.row][piece.col].color = "CHECKED"
#             return self.check_piece(board[neighbor[0]][neighbor[1]], board, player)
#         else:
#             continue

