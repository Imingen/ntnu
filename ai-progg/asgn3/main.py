import state_manager as sm














if __name__ == "__main__":

    state_manager = sm.StateManager(3)
    state_manager.init_board()
    print(state_manager.board[1][1].neighbors)
    state_manager.print_board()
    state_manager.print_hex()
    #state_manager.board[0] = [1,1,1]
    #state_manager.print_board()
    #state_manager.check_state()