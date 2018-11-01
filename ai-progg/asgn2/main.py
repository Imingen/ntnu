import game_manager as gm
import mcts as mcts


###################################################
#   This file is used to manage diffierent games
###################################################




if __name__ == "__main__":

    NUM_STICKS = 10
    MAX_STICSK_PER_TURN = 3


    # create the initial gamestate
    state = gm.GameState(NUM_STICKS, MAX_STICSK_PER_TURN)
    root = mcts.Node(None, state)
    # Should add children for the root node
    monte_carlo = mcts.MCTS()
    monte_carlo.expand_node(root)
    n = monte_carlo.traverse_tree(root)
    print(n)   
    value = monte_carlo.simulation(n)
    print(n.value)
    monte_carlo.backprop(value, n)

    print(root.value)





