import state_manager as sm
from montecarlo import MCTS
from montecarlo import Node
import matplotlib.pyplot as plt 
import matplotlib.patches as mplp 
import copy














if __name__ == "__main__":

    state_manager = sm.StateManager(8)
    state_manager.init_board()
    state_manager.print_board()
    monte_carlo = MCTS()
    print("_______________________")

##############################################

    root = Node(None, state_manager)
    lac = state_manager.get_legal_actions()
  #  print("-------------------")
    monte_carlo.expand_node(root)
    #for kid in root.children:
    #    kid.state.print_board()
    #    print("------------------")
    
    new = monte_carlo.traverse_tree(root)
    new.state.print_board()
    print("-------------------")

    m = monte_carlo.simulation(new)
    print(m)
################################################
   # print('Root value before backprop:', root.value)
   # print("Root num visits before backprop", root.num_visits)
    #monte_carlo.backprop(m, new)
   # print('Root value after backprop:',root.value)
   # print("Root num visits after backprop",root.num_visits)
    #print(len(state_manager.get_legal_actions()))




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
