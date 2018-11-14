import state_manager as sm
import matplotlib.pyplot as plt 
import matplotlib.patches as mplp 
import copy














if __name__ == "__main__":

    state_manager = sm.StateManager(5)
    state_manager.init_board()
    state_manager.print_board()
    #print(len(state_manager.get_legal_actions()))

    # a = sm.Piece(1,0,5, color="R")
    # b = sm.Piece(1,1,5, color="R")
    # c = sm.Piece(1,2,5, color="R")
    # d = sm.Piece(1,3,5, color="R")
    # e = sm.Piece(1,4,5, color="R")
    # f = sm.Piece(3,2,5, color="R")
    # g = sm.Piece(4,2,5, color="R")

    # t = sm.Piece(0,4,5, color="R")

    # print('------------------')
    # state_manager.do_move(a)
    # state_manager.do_move(b)
    # state_manager.do_move(c)
    # state_manager.do_move(d)
    #state_manager.do_move(e)
    #state_manager.do_move(f)
    #state_manager.do_move(g)
    #state_manager.do_move(t)

    # state_manager.print_board()
    # print(len(state_manager.get_legal_actions()))
    # print('------------------')
    # print('------------------')
    # print(state_manager.check_state(player=2))








    #plt.figure()
    #plt.gca().add_patch( mplp.RegularPolygon( (0.3,0.3), 6, radius=0.05, color='red' )) 
    #plt.show()