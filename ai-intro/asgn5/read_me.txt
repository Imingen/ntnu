THE SPECIFIC DATA FOR EACH SUDOKU BOARD IS ON THE PICTURES.

The easy board did not call the backtrack function at all
because the board was solved with the AC-3 algorithm alone. 

Medium and Hard seemed to do the same-ish for my solutuon with 
2 and 3 calls respectively. No failures in neither of the boards, 
because of the inference call in the backtrack algorithm, a lot of the 
possible "states" was removed which helped the backtrack algorithm and 
kept it from failing.

Very hard was called a lot more compared to the other boards and also
took a while longer to complete. This is because the inference algorithm
couldnt remove enough states on the initial call and didnt help the backtrack
algorithm during the backtracking as much as for the other boards. This also 
made the backtrack algorithm fail more than the other times as well. 

