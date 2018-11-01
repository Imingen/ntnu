import game_manager as gm
import math
import random
import copy

##################################################################
#   This file contains all logic needed for the mcts algorithm
##################################################################

class Node():
    '''
    This class contains all information for each node in the search tree:
    The state of the game in that node, the Q values, the child nodes, parent node etc
    '''
    
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
                if node.player_num == 1:
                    if self.tree_policy(child) > self.tree_policy(winner):
                        winner = child
                elif node.player_num == 2:
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
        for action in legal_actions:
            new_state = node.state.gen_successor(action)
            # Create a new node with this new state and add this as a child to the input node
            new_node = Node(node, new_state, player_num=3-node.player_num)
            # Add this new node as child for this node
            node.children.append(new_node)


    def simulation(self, node):
        '''
        Determine the value of leaf node by running a simulation of the game
        As long as there is no winner, take a random action until a terminal node is reach e.g
        the end of the game is reached
        RETURNS: The score at the end of the game: -10 if player2 wins & +10 if player1 wins
        '''

        n = copy.copy(node)
        while not n.state.is_winner():
            legal_actions = n.state.get_legal_actions()
           # print(legal_actions)
           # print("num pieces: " + str(n.state.num_pieces))
            r = random.randint(0, len(legal_actions) - 1)
         #   print("R: " + str(r))
            new_state = n.state.gen_successor(legal_actions[r])
            new_node = Node(n, new_state, player_num=3-n.player_num)
            n = new_node
         #   print(n.state.is_winner())
            if n.state.is_winner():
                #print("WINNER: " + str(n.player_num))
                break
        
        # Check what player won in this node
        if n.player_num == 1:
            return 10
        else: 
            return -10

    
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
            return avrg + (C*math.sqrt(math.log(node.parent.num_visits) / node.num_visits))
        else:
            return 0



    def get_action(self, node):
        '''
        Get an action running MCTS from a node. The node as input will be the
        root node of that search. 
        '''

        # Currently hardcoded amount of rollouts/simulations during development
        # Also called the M value in the assignment specification
        NUM_ROLLOUTS = 10
        root = node
        
        # run the game loop forever until a win/loss state is achieved
        # this should be expanded in future to have a timer so that it doesn't go on forever
      #  while True:
            # TODO: Save the search tree from MCTS rollouts, so that we dont have to compute the tree anew for every new root node







