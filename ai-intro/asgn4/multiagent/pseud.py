minimax(root, 4, True)


def minimax(node, depth, isMaxPlayer): //returns terminal value, return best value for max and min

//base case. In order to stop recursion
   if depth == 0 or node is a terminal node:
      return the heuristic value of node

   if isMaxPlayer:
      bestValue = -infinity #worst case for max player
      for each cild of the node:
          #Starting in max node, means that child nodes are MIN nodes. 
          val = minimax(child, depth -1, False) 
          bestValue = max(bestValue, val)
    return bestValue

   elif not isMaxPlayer:
      bestValue = infinity #worst case for min player
      for each child of the node:
          val = minimax(child, depth -1, True )
          bestValue = min(bestValue, val)
    return bestValue


        def minimax(node, depth, isMaxPlayer, ghost_index = 1):
          m_depth = depth
          num_ghosts = gameState.getNumAgents() - 1
          ghost_index = ghost_index

          # In order to stop the recursion
          if m_depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

          # Meaning that the pacman is making a turn
          if isMaxPlayer:
            # The worst possible value for the pacman
            bestValue = -(float("inf"))	
            # Get all the legal actions for the pacman at the current gamestate
            legalActions = gameState.getLegalActions()
            for action in legalActions:
              state = gameState.generateSuccessor(0, action)
              val = minimax(state, m_depth-1, False)
              bestValue = max(bestValue, val)
          
            return bestValue
      
          if not isMaxPlayer:
            bestValue = float("inf")
            # Legal action of the first ghost
            legalActions = gameState.getLegalActions(ghost_index)
            # Go through action for each ghost until it is the last ghost, then move to
            # pacmans turn
            if ghost_index < num_ghosts:
              # Do one move with the ghost, then compare that state to legal actions of next ghost
              for action in legalActions:
                state = gamestate.generateSuccessor(ghost_index, action)
                val = minimax(state, m_depth-1, False, ghost_index+1)
                bestValue = min(val, bestValue)
            else:
              for action in legalActions:
                state = gamestate.generateSuccessor(ghost_index, action)
                val = minimax(state, m_depth-1, True)
                bestValue = min(val, bestValue)
            return bestValue

