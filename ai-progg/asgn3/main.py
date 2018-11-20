import state_manager as sm
from montecarlo import MCTS
from montecarlo import Node
import copy


def get_action(node, num_rollouts):
    '''
    Takes in a node and runs monte-carlo-tree search 
    from this node and returns an action that would be taken from this 
    node to maximize chance of winning
    '''
    rollouts = num_rollouts
    monte_carlo = MCTS()
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
            
    best_legal = None
    for legal in root.state.get_legal_actions():
        if legal not in best.state.get_legal_actions():
            best_legal = legal

    #print([x for x in root.state.get_legal_actions() if x not in best.state.get_legal_actions()])
    return best, best_legal 

def hex_sim(M, G, board_size = 3, player=1, verbose=False):
    '''
    Run the HEX simulator using MCTS with different parameters
    '''

    BOARD_SIZE = board_size
    NUM_SIMULATIONS = M
    NUM_GAMES = G
    monte_carlo = MCTS()
    one = 0
    while NUM_GAMES > 0:
        state = sm.StateManager(BOARD_SIZE)
        state.init_board()
        root = Node(None, state)
        # if player == 2:
        #     root.player_num = 2
        # if player == 3:
        #     i = random.randint(1,2)
        #     root.player_num = i
        if verbose:
            print("\n#######################################")   
            print(f"Game #{G - (NUM_GAMES - 1)} starting")
            print("#######################################")    

        while True:
            if verbose:
                print(f"Player{root.player_num}'s turn. There is {root.state.num_pieces} sticks on the table")
                print(f"Available actions {[i for i in root.state.get_legal_actions()]}")
            new_root, a = get_action(root, NUM_SIMULATIONS)
            state.do_move(a, root.player_num)
            state.print_board()
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

    # state_manager = sm.StateManager(4)
    # state_manager.init_board()
    # state_manager.print_board()
    # monte_carlo = MCTS()
    # root = Node(None, state_manager)

    hex_sim(M=500, G=1, board_size=4)
    #action,a = get_action(root, 1000)
    #print(a)
    #print("-----------------------")
    #action.state.print_board()

    # state_manager.do_move([0,0], 2)
    # state_manager.do_move([0,1], 1)
    # state_manager.do_move([0,2], 1)
    # state_manager.do_move([1,0], 1)
    # state_manager.do_move([1,1], 2)
    # state_manager.do_move([1,2], 1)
    # state_manager.do_move([2,0], 2)
    # state_manager.do_move([2,1], 1)
    # state_manager.do_move([2,2], 2)
    # state_manager.print_board()
    # print(state_manager.check_player1_win())
    # print(state_manager.check_player2_win())









##############################################
#     root = Node(None, state_manager)
#     lac = state_manager.get_legal_actions()
#   #  print("-------------------")
#     monte_carlo.expand_node(root)
#     #for kid in root.children:
#     #    kid.state.print_board()
#     #    print("------------------")
    
#     new = monte_carlo.traverse_tree(root)
#     new.state.print_board()
#     print("-------------------")

#     m = monte_carlo.simulation(new)
#     print(m)
################################################





#######################################################
#   Code Graveyard
#######################################################


    # state_manager.do_move([1,0], 1)
    # state_manager.do_move([1,1], 1)
    # state_manager.do_move([1,2], 1)
    # state_manager.do_move([1,3], 1)
    # state_manager.do_move([1,4], 1)
    # state_manager.do_move([1,5], 1)
    # state_manager.do_move([2,5], 1)
    # state_manager.do_move([3,5], 1)
    # state_manager.do_move([3,5], 1)
    # state_manager.do_move([3,6], 1)
    # state_manager.do_move([0,1], 1)

    # state_manager.do_move([2,1], 1)
    # state_manager.do_move([3,1], 1)
    # state_manager.do_move([4,1], 1)
    # state_manager.do_move([5,1], 1)
    # state_manager.do_move([6,1], 1)
    # state_manager.do_move([7,1], 1)



    #########################################
    # EPIC CHECK 
    ##########################################
    # state_manager.do_move([0,0], 1)
    # state_manager.do_move([0,2], 1)
    # state_manager.do_move([0,4], 1)
    # state_manager.do_move([1,1], 1)
    # state_manager.do_move([1,2], 1)
    # state_manager.do_move([1,3], 1)
    # state_manager.do_move([1,5], 1)
    # state_manager.do_move([2,4], 1)
    # state_manager.do_move([3,0], 1)
    # state_manager.do_move([3,4], 1)
    # state_manager.do_move([4,0], 1)
    # state_manager.do_move([4,1], 1)
    # state_manager.do_move([4,4], 1)
    # state_manager.do_move([4,5], 1)
    # state_manager.do_move([5,3], 1)
    # state_manager.do_move([5,4], 1)


    # state_manager.do_move([0,1], 2)
    # state_manager.do_move([1,4], 2)
    # state_manager.do_move([2,1], 2)
    # state_manager.do_move([2,2], 2)
    # state_manager.do_move([2,3], 2)
    # state_manager.do_move([2,5], 2)
    # state_manager.do_move([3,1], 2)
    # state_manager.do_move([3,2], 2)
    # state_manager.do_move([3,5], 2)
    # state_manager.do_move([4,2], 2)
    # state_manager.do_move([4,3], 2)
    # state_manager.do_move([5,0], 2)
    # state_manager.do_move([5,1], 2)
    # state_manager.do_move([5,2], 2)
    # state_manager.do_move([5,5], 2)

    # state_manager.print_board()
    # print(state_manager.check_player1_win())
    # print(state_manager.check_player2_win())
    # print("DO KILlING MOVE")
    # state_manager.do_move([0,5], 2)
    # state_manager.do_move([2,0], 2)
    # state_manager.print_board()
    # print(state_manager.check_player1_win())
    # print(state_manager.check_player2_win())
