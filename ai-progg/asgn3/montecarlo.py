
############################################
#   This file contains the logic for 
#   the monte-carlo-tree-search algorithm 
############################################


class Node():
    """This class contains all information for each node in the search tree:
    The state of the game in that node, the Q values, the child nodes, parent node etc
    """
    
    def __init__(self, parent, state, player_num=1):
        self.parent = parent
        self.state = state
        self.children = []
        self.value = 0
        self.num_visits = 0
        self.player_num = player_num

    def has_children(self):
        if len(self.children) != 0:
            return True
    
    def is_root(self):
        if self.parent is None:
            return True