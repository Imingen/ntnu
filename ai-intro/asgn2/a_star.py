from collections import deque
import numpy as np
from tkinter import *

################
# The A* algorithm that I have implemented is heavily inspired by these 3 sources:
#   https://en.wikipedia.org/wiki/A*_search_algorithm
#   https://www.youtube.com/watch?v=aKYlikFAV4k
#   https://github.com/CodingTrain/AStar
# 
# I have not "copy/pasted" the code, but written it and implemented it myself. But 
# with help from the above sources
###############

class Node():
    """
    This class represents one node on the board e.g ONE cell.
    This makes it easier to keep track of coordinates, symbols, 
    f,g,h values, neighbors and previous node i.e the parent node
    """
    def __init__(self, x, y, symbol):
        self.x = x  
        self.y = y
        self.symbol = symbol
        self.f = 0
        self.g = 0
        self.h = 0
        self.neighbors = []
        self.previous = None

    def __str__(self):
        return self.symbol 
    
    def add_neighbors(self, board):
        numcols = len(board[0]) # The width of the board is the number of columns. Can be any instance, use 0 cuz why not
        numrows = len(board) # Number of rows is number of lines on the game board. 
        # These next checks is just to make sure that we dont add any neighbors outside the game space. 
        # If check is OK, add a neighbor in that directio (North, South, West or East)
        if self.y >= 1:
            north = board[self.y-1][self.x]
            self.neighbors.append(north)
        if self.y < numrows-2:
            # Use minus 2 cuz there is an empty line at the end of all the boards that I didnt take into account when loading 
            south = board[self.y + 1][self.x]
            self.neighbors.append(south)
        if self.x >= 1:
            west = board[self.y][self.x-1]
            self.neighbors.append(west)
        if self.x < numcols-1:
            east = board[self.y][self.x+1]
            self.neighbors.append(east)

# Dictionary with Node-symbol <--> color of node relationship
# Used in visualization
color_dict = {
            "▪": "yellow",
            "A": "deep pink",
            "B": "deep pink",
            ".": "bisque2",
            "#": "slate gray"}

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
    # Path to the board
    path = "./boards/"+board
    text_file = open(path, "r")
    # Reads all the lines from the file and store each line in the tmp array
    temp = text_file.read().split('\n')
    lines = []
    # This turns each line into an array of symbols and appends that list to the array called lines
    for line in temp:
        wlist = list(line)
        lines.append(wlist)
    # Now exists a 2D array of lines and each lines is an array of the symbols on that line
    return lines

def make_board_into_nodes(arr):
    """
    Turns 2D array of symbols into a 2D array of Node objects
    This will be used as the board in the A* search algorithm 
    Input: A 2D array of symbols
    """
    board = arr
    # Loops through all symbols in the 2D array and makes a Node object where the indexes is the x,y values
    # and stores the symbol in the Node object. Adds the Node object to a board list which stores all the Nodes. 
    for i, line in enumerate(arr):
        for j, symbol in enumerate(line):
            node = Node(j, i, symbol)
            board[i][j] = node
    
    # Adds all the neighbors for each node
    # I don't do this in the same loop as above cuz I want to make sure all the nodes 
    # created before I set up neighbor relationship cuz those nodes might not be created yet. 
    for i, line in enumerate(board):
        for j, node in enumerate(line):
            node.add_neighbors(board)
    # return the new board, ready to be inputted to the A* algorithm        
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
    The A* algorithm. 
    Input: Starting node, end node(goal state) and the board (board of nodes). 
    This algorithm doesnt return the path but draws it at the end. 
    """
    # The board (game space)
    board_n = board
    # Set of nodes evaluated
    closed_set = deque()
    # Currently discovered nodes, but not evaluated
    # Initialized with the starting node
    open_set = deque()
    open_set.append(start)

    # The shortest path. Will hold all the nodes that is in the final path
    path = list()

    # As long as there is nodes in the open set, search for a goal
    while open_set:
        # Current node = the node in the open set with the lowest f score
        # To find the lowest f score, I just set it to the first index in the open set
        # and loop through the open set to check if there is anyone with lower f score
        lowest_f_index = 0
        for i, node in enumerate(open_set):
            if open_set[i].f < open_set[lowest_f_index].f:
                lowest_f_index = i

        #Sets the current node to the node in the open set with the lowest f score
        current_node = open_set[lowest_f_index]
        # FIRST! Check if this node actually is the end
        if current_node == end:
            # If this is the goal we backtrack through the previous nodes
            # and add each node to the path 
            tmp = current_node
            path.append(tmp)
            while tmp.previous:
                path.append(tmp.previous)
                tmp = tmp.previous
            # Now all nodes in the shortest path are stored in a variable
            # and I will visualize the result
            for elem in path[1:-1]: #Not redrawing the A and B as I want to see where start and finnish is
                elem.symbol = '▪' # Change the symbol of the Node in the path so it is easy to see where the path is drawn
            # Draw the new board
            draw_4k(board_n, wait = True)
            # Algorithm done. 

        # Remove the current node from the open set
        open_set.remove(current_node)
        # And add the current node to the closed set
        closed_set.append(current_node)

        # Get all the neighbors to the current node
        neighbors = current_node.neighbors
        # And start looping through all the neighbors
        for nb in neighbors:
            # If it is in the closed set it is allready evaluated so we ignore this neighbor
            # Also if this neighbor has a wall symbol we ignore it (easy wall check )
            if nb in closed_set or nb.symbol is "#":
                continue
            
            # tentative g score to be used in later chekcs. 
            # also called the distance from start to the neighbor
            tmp_g = current_node.g + 1 #Each step is increased by one

            # add the neigbor to the open set so we can eval it 
            # when we reach it
            if nb not in open_set:
                open_set.append(nb)

            #If the tentative score is higher than the neighbors g score we ignore that neighbor
            elif tmp_g >= nb.g:
                continue

            #Adds this current node as the neighbors previous e.g the parent node
            nb.previous = current_node       
            # Adds the tmp g score to the neighbors g score since it is lower
            nb.g = tmp_g     
            # Calculate heuristic
            nb.h = calculate_manhattan(nb, end)
            # Add the g score and heuristic together to the neighbors f function
            nb.f = nb.g + nb.h
        

def draw_4k(board, wait = False):
    """
    Method that visualizes the board using TkInter
    Input: The board to be drawn and a flag variable to be used to continue
    or to stop so the windows dont dissapear
    """
    root = Tk()
    for i, line in enumerate(board):
        for j, symbol in enumerate(line):
            # Creates a label (tkinter object)
            # each labels will show the symbol
            label = Label(root, text = symbol.symbol)
            # each labels gets the correct color from the color dict
            label.config(bg=color_dict[symbol.symbol])
            # Add the label to the GUI with the same coordinates as the indixe of the Node on the board
            label.grid(row=i, column=j,  sticky="nsew")
    # Run this on last show so the windows doesnt dissapear
    if wait:
        root.mainloop()
    else:
        # This will continue the UI loop so we can see both starting and end window to compare
        root.update_idletasks()
        root.update()

def main():
    # Load the board from textual representation into an array
    board_text = board_loader("board-1-4.txt")
    # Make the board into a  board of Node objects
    board = make_board_into_nodes(board_text)    
    # Find the start and end nodes
    start, end = find_start_and_end(board)
    # Draw the board before path is found
    draw_4k(board)
    # Run the A* algorithm with the inputs found above
    a_star(start, end, board)



if __name__ == '__main__':
    main()

   