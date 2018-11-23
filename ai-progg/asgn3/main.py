import state_manager as sm
from montecarlo import MCTS
from montecarlo import Node
from singularity import NeuralNet
import copy
import time
import numpy as np 
import random
import pickle
from topp import TOPP


def get_action(node, num_rollouts, net, neural_net, epsilon = 0.1):
    '''
    Takes in a node and runs monte-carlo-tree search 
    from this node and returns an action that would be taken from this 
    node to maximize chance of winning
    '''
    rollouts = num_rollouts
    monte_carlo = MCTS()
    root = node
    anet = neural_net
    
    while rollouts > 0:
        rollouts -= 1
        if len(root.children) == 0:
            monte_carlo.expand_node(root)
        n = monte_carlo.traverse_tree(root)
        if n.num_visits == 0:
            value = monte_carlo.simulation(n, anet, net, epsilon)
            monte_carlo.backprop(value, n)
            continue
        if n.num_visits > 0:
            monte_carlo.expand_node(n)
            if len(n.children) > 0:
                nn = n.children[0]
                value = monte_carlo.simulation(nn, anet, net, epsilon)
                monte_carlo.backprop(value, nn)
            else:
                if n.player_num == 1:
                    monte_carlo.backprop(-1, n)
                else:
                    monte_carlo.backprop(1, n)

    # Find the child of root with the most visit counts
    best = root.children[0]
    for child in root.children:
        if child.num_visits > best.num_visits:
            best = child
    # Returns row,col as action to take in the main game tree            
    best_legal = None
    for legal in root.state.get_legal_actions():
        if legal not in best.state.get_legal_actions():
            best_legal = legal

    case = root.create_training_case()

    #print([x for x in root.state.get_legal_actions() if x not in best.state.get_legal_actions()])
    return best, best_legal, case

def hex_sim(M, G, net=None, epsilon_decay=0.05, epsilon_begin=1, board_size = 3, player=1, 
                mb_size = 28, verbose=False, save_interval=50):
    '''
    Run the HEX simulator using MCTS with different parameters
    '''

    BOARD_SIZE = board_size
    NUM_SIMULATIONS = M
    NUM_GAMES = G
    monte_carlo = MCTS()
    one = 0

    replay_buffer = []
    anet = net.get_anet()
    i = save_interval
    epsilon = epsilon_begin

    while NUM_GAMES > 0:
        if epsilon > 0.1:
            epsilon = epsilon - epsilon_decay
        if epsilon < 0.1:
            epsilon = 0.1

        state = sm.StateManager(BOARD_SIZE)
        state.init_board()
        root = Node(None, state)

        if player == 2:
            root.player_num = 2
        if player == 3:
            q = random.randint(1,2)
            root.player_num = q

        if NUM_GAMES % i == 0 or G == NUM_GAMES:
            path = MODELS_SAVE_PATH + str((G - NUM_GAMES)) +".h5"
            anet.save(path)  

        if verbose:
            print("\n#######################################")   
            print(f"Game #{G - (NUM_GAMES - 1)} starting")
            print("#######################################")    

        while True:

            new_root, a, training_case = get_action(root, NUM_SIMULATIONS, anet,net, epsilon=epsilon)
            replay_buffer.append(training_case)
            print(f"ACTION: {a}, for player{root.player_num}")
            state.do_move(a, root.player_num)
            state.print_board_pretty()
            #if verbose:
                #print(f"Player{root.player_num}'s turn. There is {root.state.num_pieces} sticks on the table")
             #   print(f"Available actions {[i for i in state.get_legal_actions()]}")

            if state.is_winner():
                break
            root = new_root
            root.parent = None
            #root.player_num = new_root.player_num
            #root.player_num = new_root.player_num

 
        print(f"Player {root.player_num} won game #{G - (NUM_GAMES - 1)} ")
        print(f"Replay BUFFER: {len(replay_buffer)}")
        mb_size = mb_size
        t = len(replay_buffer)
        k = mb_size if t >= mb_size else len(replay_buffer)
        mbatch = random.sample(replay_buffer, k)
        nn.train_anet(anet, mbatch)
        

        if root.player_num == 1:
           one += 1
 
        NUM_GAMES -= 1

    # Save the last version of the model 
    path = MODELS_SAVE_PATH + str((G - NUM_GAMES)) +".h5"
    anet.save(path) 
        
    print("\n#######################################")    
    print(f"Player 1 won {one} out of {G} games")
    print("#######################################")    

if __name__ == "__main__":


    #################################################
    #   MAIN ALGORITHM PARAMETERS
    #################################################
    RBUFF_SAVE_PATH = "rbuff_3x3"
    MODELS_SAVE_PATH = "/home/marius/ntnu/ai-progg/asgn3/models_2/"
    BOARD_SIZE = 3
    SAVE_INTERVAL = 5
    NUM_ROLLOUTS = 20
    NUM_GAMES = 200
    EPSILON_DECAY = 0.02
    EPSILON_BEGIN = 0.1
    MB_SIZE = 512
    #################################################
    #   NEURAL NETWORK PARAMETERS
    #################################################
    ACTIVATION = 'relu'
    OUTPUT_ACTIVATION = 'softmax'
    LOSS = 'mean_squared_error'
    OPTIMIZER = 'adam'
    LEARNING_RATE = 0.01
    NUMBER_HIDDEN_LAYERS = 3
    NEURONS_IN_HIDDEN = [200, 200]
    DROP_OUT = True
    DROP_OUT_RATE = 0.2
    NUM_INPUT = BOARD_SIZE**2
    SPLIT = 0.8
    EPOCHS = 5
    ###################################################
    #   TOPP PARAMETERS
    ##################################################
    NUM_TOPP_GAMES = 20
    PATH = "/home/marius/ntnu/ai-progg/asgn3/models_2/"
    TOPP_VERBOSE = False




    nn = NeuralNet(LEARNING_RATE, NUM_INPUT, NEURONS_IN_HIDDEN, loss=LOSS,  split=SPLIT,
                    optimizer=OPTIMIZER, activation=ACTIVATION, epochs=EPOCHS, output_activation=OUTPUT_ACTIVATION)


    t0 = time.time()

    hex_sim(M=NUM_ROLLOUTS, G=NUM_GAMES, net = nn, board_size=BOARD_SIZE, verbose=True, 
          epsilon_decay=EPSILON_DECAY, epsilon_begin=EPSILON_BEGIN,player=3,
          mb_size=MB_SIZE, save_interval=SAVE_INTERVAL)
    
    t1 = time.time()
    print(f"TIME USED: {t1 - t0}")

    # COMMENT HERE IF TOPP SHOULD BE RUN
    #topp = TOPP(net_man = nn,  num_games=NUM_TOPP_GAMES, board_size=BOARD_SIZE, path=PATH)
    #topp.run_tournament(verbose=TOPP_VERBOSE)












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
