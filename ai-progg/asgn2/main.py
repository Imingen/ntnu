import game_manager as gm
import mcts as mcts
import pdb


###################################################
#   This file is used to manage different games
###################################################


def get_action(node, num_rollouts):
    '''
    Takes in a node and runs monte-carlo-tree search 
    from this node and returns an action that would be taken from this 
    node to maximize chance of winning
    '''
    x = num_rollouts
    monte_carlo = mcts.MCTS()
    root = node
    while x != 0:
        if len(root.children) == 0:
            monte_carlo.expand_node(root)
       # print(f"Number of children in root: {len(root.children)}")
        n = monte_carlo.traverse_tree(root)
      # print(f"n: {n}")
        value = monte_carlo.simulation(n)
        monte_carlo.backprop(value, n)
        if n.has_children():
            monte_carlo.expand_node(n)
        x -= 1
    best = root.children[0]
    for child in root.children:
        if child.num_visits > best.num_visits:
            best = child
    return (root.state.num_pieces - best.state.num_pieces), best



if __name__ == "__main__":

    NUM_STICKS = 10
    MAX_STICSK_PER_TURN = 3
    y = 0
    playerone = 0
    while y < 50:
        print(f"Y: {y}")
        # create the initial gamestate
        state = gm.GameState(NUM_STICKS, MAX_STICSK_PER_TURN)
        root = mcts.Node(None, state)
        x = NUM_STICKS
        #get_action(root, 100)
        while x != 0:
            a, new_root = get_action(root, 50)
            root = new_root
            x -= a
            if x == 0:
                break
        

        
    # print(f"Final results: player one won {playerone} out of {y} games")

    # Should add children for the root node
    # monte_carlo = mcts.MCTS()
    # monte_carlo.expand_node(root)
    # n = monte_carlo.traverse_tree(root)
    # print(n)   
    # value = monte_carlo.simulation(root)
    # print(n.value)
    # monte_carlo.backprop(value, n)
    # print(n.value)
    # print(root.value)
    # print(root.num_visits)





