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

    legalActions = gameState.getLegalActions()
    numghosts = gameState.getNumAgents() - 1
    bestaction = Directions.STOP
    score = -(float("inf"))
    for action in legalActions:
        nextState = gameState.generateSuccessor(0, action)
        prevscore = score
        score = max(score, minvalue(nextState, self.depth, 1, numghosts))
        if score > prevscore:
            bestaction = action
return bestaction