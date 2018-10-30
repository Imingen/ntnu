import game_manager as gm
import math
import random

##################################################################
#   This file contains all logic needed for the mcts algorithm
##################################################################

class Node():
    '''
    This class contains all information for each node in the search tree:
    The state of the game in that node, the Q values, the child nodes, parent node etc
    '''
    
    def __init__(self, parent, state):
        self.parent = parent
        self.state = state
        self.children = []
        self.value = 0
        self.num_visits = 0

    def has_children(self):
        if self.children:
            return True
    
    def is_root(self):
        if self.parent is None:
            return True
        
    

class MCTS():
    '''
    This class contains all logic concerning the monte-carlo-tree-search algorithm
    '''
    # Init root and state
    initial_state = gm.GameState(10, 3)
    root = Node(None, initial_state)
    visited = [root]

    current = root


    def traverse_tree(node):
        # The current node is not a leaf node
        if node.has_children():
            # This means that the node has children and we should traverse
            # to the next child-node that maximizes the UCB1 value
            winner = node.children[1]
            for child in node.children:
                if tree_policy(child) > tree_policy(winner):
                    winner = child
            # Now we have found the child to go to next
            # Continue to find the best child-node to go to until we are at a leaf node
            traverse_tree(winner)
        # Current node is a leaf node
        elif not node.has_children():
            # We have traversed the whole tree up until a noe that has no more children 
            return node 

    
    def expand_node(node):
        '''
        Expand the leaf node we are at now
        '''
        # Get all possible actions from this node
        legal_actions = node.state.get_legal_actions()
        # Choose a random action to take
        r = random.randint(1, len(legal_actions)+1)
        # Get the new state that comes from taking that random action 
        new_state = node.state.gen_successor(legal_actions[r])
        # Create a new node with this new state and add this as a child to the input node
        new_node = Node(node, new_state)
        return new_node


    def simulation(node):
        '''
        Determine the value of leaf node by running a simulation of the game
        As long as there is no winner, take a random action until a terminal node is reach e.g
        the end of the game is reached
        '''
        


    
    def backprop():
        x = "TODO"
    

    def tree_policy(node):
        # Using the UCB1 (or UCT if you like) magic to figure out which node to expand
        C = 1
        avrg = node.value / node.num_visits
        return avrg + (C*math.sqrt(math.log(node.parent.num_visits) / node.num_visits))



    def run_search():
        '''
        Run search from the current node. This need not be the root node
        '''
        current = traverse(root)






