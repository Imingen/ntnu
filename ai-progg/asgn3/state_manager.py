import numpy as np
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
        """Prints the game board as it is represented
        internally in the system: As a n x n matrix
        """
        for i, row in enumerate(self.board):
            c = []
            for j, col in enumerate(self.board):
                c.append(self.board[i][j].color)
            print(c)
        print("____________________")

    def print_board_pretty(self):
        """Prints the board in the format the HEX game
        is represented as to the players: A diamond shape.
        """
        n = 0
        m = 0

        for i in range(0, n+1):
            for j in range(0, n+1, ):
                print(i, j)
                n = n + 1
                m = m + 1
        x = ["0", " ", "0", "0", " "]
        y = ''.join(x)
        print(y)

        # ok = self.get_flat_board()
        # index = 0
        # m = self.size
        # n = 1
        # for i in range(1, m+1):
        #     for j in range(1, (m-i)+1):
        #         print(end=" ")
        #     for i in range(0, n):
        #         print("*", end = " ")
        #     n = n + 1
        #     print()
            
        
        # for i in range(m-1, 0, -1):
        #     for j in range((m-i)+1, 1, -1):
        #         print(end=" ")

        #     for i in range(1, n-1):
        #         print("*", end = " ")
        #     n = n - 1
        #     print()
                

    def get_flat_board(self):
        """Returns a flat representions of the board. 
        To be used as input to the neural network.
        
        Returns: A 1D numpy array that represents the game-board
        """
        flat_list = []
        for i in range(0, self.size):
            for j in range(0, self.size):
                if self.board[i][j].color == self.color_dict[1]:
                    flat_list.append(1)
                if self.board[i][j].color == self.color_dict[2]:
                    flat_list.append(2)
                if self.board[i][j].color == self.color_dict[3]:
                    flat_list.append(0)
        return np.array([flat_list])


    def int_to_index(self, x):
        """Turns a simple int which is an index in a 1D array
        into an index for a matrix

        Input: x, an integer. 

        Returns: An array [i,j] of row and column in a matrix
        """
        i = x // self.size
        j = x % self.size
        return [i,j]

    def get_legal_actions(self):
        legal_actions = []
        for i in range(0, self.size):
            for j in range(0, self.size):
                if self.board[i][j].color == 'W':
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
                if result == True:
                    return result
                else:
                    continue
        return result

    def check_player2_win(self):
        result = False
        for piece in self.board:
            if piece[0].color == self.color_dict[2]:
                self.checked_nodes = []
                result = self.check_piece(piece[0], self.board, player = 2)
                if result == True:
                    return result
                else:
                    continue
        return result

    def is_winner(self):
        if self.check_player1_win() == True:
            return True
        if self.check_player2_win() == True:
            return True
        else:
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

