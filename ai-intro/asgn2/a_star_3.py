from collections import deque
import numpy as np
from tkinter import *

'''
1. Load the board into an array CHEKC
2. Find start and goal on the board CHECK
3. Implement A* (Using the wikipedia page) (https://en.wikipedia.org/wiki/A*_search_algorithm)
4. Implementing a wrapper class e.g a node class for holding neighbors and obstacles etc. 
'''

class Node():
    '''
    This class is one node on the board
    Encapsulate in class so that I can store additional information liek neighbors, symbol, x/y coords
    '''
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
        #return "x: {}, y: {}. Symbol: {}".format(self.x, self.y, self.symbol)
        return self.symbol

    def redraw_symbol(self, new_symbol):
        self.symbol = new_symbol
    
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


def find_start_and_end(arr):
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
    # Go through the board and and make a Node() out of the info in that spot
    # and store that node in a new array with the same placemenet but more information
    path = "./boards/"+board
    text_file = open(path, "r")
    # Temp = an array where each element is a line. Would like to turn this into a 2D array
    # where each element is a line and each element inside each line is a symbol on that line
    temp = text_file.read().split('\n')
    lines = []
    for line in temp:
        wlist = list(line)
        lines.append(wlist)

    return lines

def make_board_into_nodes(arr):
    '''
    takes in a 2d array and makes a new 2D array but with Node() objects on each place
    '''
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
    string = ""        
    for i, line in enumerate(arr):
        string = ""
        for j, node in enumerate(line):
            string += node.__str__()
        print(string)

def calculate_manhattan(node_a, node_b):
    return (abs(node_a.x - node_b.x) + abs(node_a.y - node_b.y))

cost_dict = {
            "A": 0,
            "B": 0,
            "#": 0,
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

def a_star(start, end, board):
    '''
    The core algorithm.
    Input: Starting node, end node(goal state) and the board. 
    '''
    board_n = board
    # Set of nodes evaluated
    closed_set = deque()
    # Currently discovered nodes, but not evaluated
    # Initialized with the starting node
    open_set = deque()
    open_set.append(start)

    path = list()


    while open_set:
        # Current node = the node in the open set with the lowest f score
        # Assuming that it is the first node so that we can update it
        lowest_f_index = 0
        for i, node in enumerate(open_set):
            if open_set[i].f < open_set[lowest_f_index].f:
                lowest_f_index = i

        current_node = open_set[lowest_f_index]
        #print(current_node.__str__())
        if current_node == end:
            
            tmp = current_node
            path.append(tmp)
            while tmp.previous:
                path.append(tmp.previous)
                tmp = tmp.previous
            print("FINNISHED")

        open_set.remove(current_node)
        closed_set.append(current_node)

        neighbors = current_node.neighbors
        for nb in neighbors:
            
            if nb in closed_set or nb.symbol is "#":
                continue
            
            tmp_g = current_node.g + nb.cost # Increas G by one, but just keep it in tmp var until more hecks

            if nb in open_set:
                if tmp_g < nb.g:
                    nb.g = tmp_g
            else:
                nb.g = tmp_g
                open_set.append(nb)

            nb.h = calculate_manhattan(nb, end)
            nb.f = nb.g + nb.h
            nb.previous = current_node
    
    for elem in path[1:-1]: #Not redrawing the A and B as I want to see where start and finnish is
        elem.symbol = '▪'
    redraw_board(board_n)
    draw_4k(board_n, wait = True)

def dijkstra():
    continue

def bfs():
    continue


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
    board_text = board_loader("board-2-3.txt")
    board = make_board_into_nodes(board_text)
    start, end = find_start_and_end(board)
    draw_4k(board)
    a_star(start,end,board)


if __name__ == '__main__':
    main()

   