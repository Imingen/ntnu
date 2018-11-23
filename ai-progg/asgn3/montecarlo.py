import copy
import random
import math 
from singularity import NeuralNet
import numpy as np
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

    def create_training_case(self):
        # Make a tuple of the root state and the distribution of the visit counts 
        # of the roots children, to be used as a training case for the ANET
        this_state = self.state.get_flat_board(self.player_num)
        children_visitcount = [child.num_visits for child in self.children]
        tmp = this_state[0]
        distribution = [0] * len(tmp[1:])
        for i, item in enumerate(tmp[1:]):
            if item == 0:
                distribution[i] = children_visitcount.pop(0)
        # print(f"distribution: {distribution}")    
        # print(f"this state: {this_state[0]}")    
        distribution = np.array(distribution)
        case = [this_state[0], distribution]
        case = tuple(case)
        return case

class MCTS():
    '''
    This class contains all logic concerning the monte-carlo-tree-search algorithm
    '''

    def traverse_tree(self, node):
        current = node
        # The current node is not a leaf node
        while current.has_children():
            # This means that the node has children and we should traverse
            # to the next child-node that maximizes the UCB1 value
            winner = current.children[0]
            for child in current.children:
                if child.num_visits == 0:
                    winner = child
                    break
                if current.player_num == 1:
                    if self.tree_policy(child) > self.tree_policy(winner):
                        winner = child
                elif current.player_num == 2:
                    if self.tree_policy(child) < self.tree_policy(winner):
                        winner = child
            current = winner
            # Now we have found the child to go to next
            # Continue to find the best child-node to go to until we are at a leaf node
            if current.has_children == False:
                break
        # Current node is a leaf node
        # We have traversed the whole tree up until a node that has no more children 
        return current

    
    def expand_node(self, node):
        '''
        Expand the leaf node we are at now
        '''
        # Get all possible actions from this node
        legal_actions = node.state.get_legal_actions()
        # Generate all the new states from these legal actions
        if legal_actions is not None:
            for action in legal_actions:
                new_state = node.state.gen_successor(action, player=node.player_num)
                # Create a new node with this new state and add this as a child to the input node
                new_node = Node(node, new_state, player_num=3-node.player_num)
                # Add this new node as child for this node
                node.children.append(new_node)


    def simulation(self, node,nn, model, epsilon):

        n = copy.copy(node)
        if n.state.check_player1_win():
            return 1
        if n.state.check_player2_win():
            return 0
            
        while True:
            if n.player_num == 1:
                result = n.state.check_player1_win()
                if result is True:
                    # print('Player ONE')
                    # n.state.print_board()
                    #n.state = new_state
                    n.state.winner = "Player ONE"
                    break
            if n.player_num == 2:
                result = n.state.check_player2_win()
                if result is True:
                    # print('Player TWO')
                    # n.state.print_board()
                    #n.state = new_state
                    n.state.winner = "Player TWO"
                    break
            
            try:
                action = nn.neural_magic(model, node.state, node.player_num, epsilon=epsilon)
            except ValueError:
                print(n.state)
                print(n.state.get_flat_board())
           # print(index)
           # print(action)
            new_state = n.state.gen_successor(action, n.player_num)
            #new_state.print_board()
            new_node = Node(n, new_state, player_num=3-n.player_num)
            n = new_node
        #n.state.print_board()
        #print(f"Winner: {n.state.winner}")

        if n.player_num == 1:
            return 1
        else: 
            return -1

    
    def backprop(self, terminal_value, node):
        '''
        Take the value from the simulation and backpropagate it upwards in the tree
        and update the values. 
        Input: The value of the terminal node & the node that we begin the backpropagating on.
        '''
        n = node
        while n.parent is not None:
            n.num_visits += 1
            n.value += terminal_value
            n = n.parent
        if n.parent is  None:
            n.num_visits += 1
            n.value += terminal_value


    def tree_policy(self, node):
        # Using the UCB1 (or UCT if you like) magic to figure out which node to expand
        C = 1
        if node.num_visits is not 0:
            avrg = node.value / node.num_visits
            if node.parent.player_num == 1:
                return avrg + (C*math.sqrt(math.log(node.parent.num_visits) / node.num_visits))
            elif node.parent.player_num == 2:
                return avrg - (C*math.sqrt(math.log(node.parent.num_visits) / node.num_visits))
        else:
            return 0














