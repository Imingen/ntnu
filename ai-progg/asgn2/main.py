import game_manager as gm
import mcts as mcts
import pdb
import random


###################################################
#   This file is used to manage different games
###################################################


def get_action(node, num_rollouts):
    '''
    Takes in a node and runs monte-carlo-tree search 
    from this node and returns an action that would be taken from this 
    node to maximize chance of winning
    '''
    rollouts = num_rollouts
    monte_carlo = mcts.MCTS()
    root = node
    while rollouts > 0:
        rollouts -= 1
        if len(root.children) == 0:
            monte_carlo.expand_node(root)
        n = monte_carlo.traverse_tree(root)
        if n.num_visits == 0:
            value = monte_carlo.simulation(n)
            monte_carlo.backprop(value, n)
            continue
        if n.num_visits > 0:
            monte_carlo.expand_node(n)
            if len(n.children) > 0:
                nn = n.children[0]
                value = monte_carlo.simulation(nn)
                monte_carlo.backprop(value, nn)
            else:
                if n.player_num == 1:
                    monte_carlo.backprop(-1, n)
                else:
                    monte_carlo.backprop(1, n)

    best = root.children[0]
    for child in root.children:
        if child.num_visits > best.num_visits:
            best = child
    return (root.state.num_pieces - best.state.num_pieces), best

def nim_sim(N, K, M, G, player=1, verbose=False):
    '''
    Run the NIM simulator using MCTS with different parameters
    '''
    NUM_STICKS = N
    MAX_STICSK_PER_TURN = K
    NUM_SIMULATIONS = M
    NUM_GAMES = G
    monte_carlo = mcts.MCTS()
    one = 0
    while NUM_GAMES > 0:
        state = gm.GameState(NUM_STICKS, MAX_STICSK_PER_TURN)
        root = mcts.Node(None, state)
        if player == 2:
            root.player_num = 2
        if player == 3:
            i = random.randint(1,2)
            root.player_num = i
        if verbose:
            print("\n#######################################")   
            print(f"Game #{G - (NUM_GAMES - 1)} starting")
            print("#######################################")    

        while True:
            if verbose:
                print(f"Player{root.player_num}'s turn. There is {root.state.num_pieces} sticks on the table")
                print(f"Available actions {[i for i in root.state.get_legal_actions()]}")
            a, new_root = get_action(root, 60)
            state.do_move(a)
            if verbose:
                print(f"Player {root.player_num} took {a} number of stones")
                print(f"There is {state.num_pieces} stones left \n")
            if state.is_winner():
                break
            root = new_root
        print(f"Player {root.player_num} won game #{G - (NUM_GAMES - 1)} ")
        if root.player_num == 1:
           one += 1
        NUM_GAMES -= 1
    print("\n#######################################")    
    print(f"Player 1 won {one} out of {G} games")
    print("#######################################")    




if __name__ == "__main__":

    nim_sim(N=7, K=2, M=1000, G=50, player=1, verbose=True)









########################################################################
#               CODE GRAVEYARD RIPERONI PEPPERONI 
########################################################################
    # NUM_STICKS = 10
    # MAX_STICSK_PER_TURN = 3
    # state = gm.GameState(NUM_STICKS, MAX_STICSK_PER_TURN)
    # monte_carlo = mcts.MCTS()
    # root = mcts.Node(None, state)
    # monte_carlo.expand_node(root)
    # current = monte_carlo.traverse_tree(root)
    # for i in range(1, 11):
    #     monte_carlo.expand_node(current)
    #     value = monte_carlo.simulation(current)
    #     monte_carlo.backprop(value, current)
    #     current = monte_carlo.traverse_tree(root)
    
    # X = 2
    # while True:
    #     X += 1


    
    # NUM_STICKS = 7
    # MAX_STICSK_PER_TURN = 2
    # monte_carlo = mcts.MCTS()
    # one = 0
    # y = 100
    # x = y
    # while x > 0:
    #     state = gm.GameState(NUM_STICKS, MAX_STICSK_PER_TURN)
    #     root = mcts.Node(None, state)
    #    # print(f"State: number of sticks: {state.num_pieces}. \n")

    #     while True:
    #         #print(f"Player{root.player_num}'s turn. There is {root.state.num_pieces} sticks on the table")
    #       #  print(f"Available actions {[i for i in root.state.get_legal_actions()]}")
    #         a, new_root = get_action(root, 60)
    #        # print(f"Player {root.player_num} took {a} number of stones ")
    #         state.do_move(a)
    #         if state.is_winner():
    #             break
    #       #  print(f"Player{root.player_num} took {a} number of sticks. Number of sticks left on the table: {state.num_pieces}. \n")
    #         #new_num = 3 - root.player_num
    #         #root = mcts.Node(None, state)
    #         root = new_root
    #         #root.player_num = new_num
    #     #print(f"Final state: {root.state.num_pieces}")
    #     if root.player_num == 1:
    #         one += 1
    #     x -= 1
    # print(one)




