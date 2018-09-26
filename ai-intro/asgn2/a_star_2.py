from collections import deque
import numpy as np
from tkinter import *

################
# This is the same as a_star.py just for the 2nd part of the assignment
# The same comments apply from the previous files where the code is the same. Additional 
# comments are added where there is new code
###############

class Node():
    """
    This class represents one node on the board e.g ONE cell.
    This makes it easier to keep track of coordinates, symbols, 
    f,g,h values, neighbors and previous node i.e the parent node
    """
    def __init__(self, x, y, symbol, cost):
        self.x = x
        self.y = y
        self.symbol = symbol
        self.f = 0
        self.g = 0
        self.h = 0
        self.cost = cost
        self.neighbors = []
        self.previous = None

    def __str__(self):
        return self.symbol

    def add_neighbors(self, board):
        numrows = len(board[0])
        numcols = len(board)
        if self.y >= 1:
            north = board[self.y-1][self.x]
            self.neighbors.append(north)
        if self.y < numcols-2:
            # Use minus 2 cuz there is an empty line at the end of all the boards that I didnt take into account when loading 
            south = board[self.y + 1][self.x]
            self.neighbors.append(south)
        if self.x >= 1:
            west = board[self.y][self.x-1]
            self.neighbors.append(west)
        if self.x < numrows-1:
            east = board[self.y][self.x+1]
            self.neighbors.append(east)

# A dictionary with the symbol <--> cost relationship
cost_dict = {
            "A": 0,
            "B": 0,
            "w": 100,
            "m": 50,
            "f": 10,
            "g": 5,
            "r": 1}
color_dict = {
            "▪": "yellow",
            "A": "deep pink",
            "B": "deep pink",
            "w": "SteelBlue1",
            "m": "ivory4",
            "f": "SpringGreen4",
            "g": "SeaGreen1",
            "r": "slate gray"}

def find_start_and_end(arr):
    """
    Finds the start and end node.
    Will loop through and array and find the Nodes with the symbol A and one with the symbol B
    """
    start = None
    end = None
    for i, line in enumerate(arr):
        for j, node in enumerate(line):
            if node.symbol is 'A':
                start = node
            if node.symbol is 'B':
                end = node
    return start, end
                
def board_loader(board):
    """
    Loads the board into an array.
    Input: Name of the file e.g a .txt file representing the board game
    """
    path = "./boards/"+board
    text_file = open(path, "r")
    temp = text_file.read().split('\n')
    lines = []
    for line in temp:
        wlist = list(line)
        lines.append(wlist)

    return lines

def make_board_into_nodes(arr):
    """
    Turns 2D array of symbols into a 2D array of Node objects
    This will be used as the board in the A* search algorithm 
    Input: A 2D array of symbols
    """
    board = arr
    for i, line in enumerate(arr):
        for j, symbol in enumerate(line):
            node = Node(j, i, symbol, cost_dict[symbol])
            board[i][j] = node
    
    for i, line in enumerate(board):
        for j, node in enumerate(line):
            node.add_neighbors(board)
    return board

def redraw_board(arr):
    """
    This function was used to print the new board with updated node symbols
    to represent the path to the console. But I changed to GUI visualization later in 
    development
    Just turns a board of nodes into a textual representation again and prints each line 
    to console for visualisation. 
    """ 
    string = ""        
    for i, line in enumerate(arr):
        string = ""
        for j, node in enumerate(line):
            string += node.__str__()
        print(string)

def calculate_manhattan(node_a, node_b):
    """
    Calculates the Manhattan distance between two nodes
    """
    return (abs(node_a.x - node_b.x) + abs(node_a.y - node_b.y))



def a_star(start, end, board):
    """
    The A* algorithm. THIS VERSION INCLUDES DIFFERENT COSTS PER NODE
    Input: Starting node, end node(goal state) and the board (board of nodes). 
    This algorithm doesnt return the path but draws it at the end. 
    """
    board_n = board
    closed_set = deque()
    open_set = deque()
    open_set.append(start)

    path = list()

    while open_set:
        lowest_f_index = 0
        for i, node in enumerate(open_set):
            if open_set[i].f < open_set[lowest_f_index].f:
                lowest_f_index = i
            # Adds an additional check in case the f values are similar. Then we compare the g score instead
            # and find the lowest
            if open_set[i].f == open_set[lowest_f_index].f:
                if open_set[i].g < open_set[lowest_f_index].g:
                    lowest_f_index = i

        current_node = open_set[lowest_f_index]

        if current_node == end:
            tmp = current_node
            path.append(tmp)
            while tmp.previous:
                path.append(tmp.previous)
                tmp = tmp.previous
            for elem in path[1:-1]: 
                elem.symbol = '▪'
            draw_4k(board_n, wait = True)

        open_set.remove(current_node)
        closed_set.append(current_node)

        neighbors = current_node.neighbors
        for nb in neighbors:
            if nb in closed_set: #Doesnt check walls here since there is no walls
                continue
            
            tmp_g = current_node.g + nb.cost # Adds the cost of the neighbor cell to the tentative g score instead of just 1

            if nb not in open_set:
                open_set.append(nb)
          
            elif tmp_g >= nb.g:
                continue

            nb.previous = current_node       
            nb.g = tmp_g     
            nb.h = calculate_manhattan(nb, end)
            nb.f = nb.g + nb.h

def draw_4k(board, wait = False):
    root = Tk()
    for i, line in enumerate(board):
        for j, symbol in enumerate(line):
            label = Label(root, text = symbol.symbol)
            label.config(bg=color_dict[symbol.symbol])
            label.grid(row=i, column=j,  sticky="nsew")
    if wait:
        root.mainloop()
    else:
        root.update_idletasks()
        root.update()

def main():
    board_text = board_loader("board-2-4.txt")
    board = make_board_into_nodes(board_text)
    start, end = find_start_and_end(board)
    draw_4k(board)
    a_star(start,end,board)


if __name__ == '__main__':
    main()

   