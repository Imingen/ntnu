#!/usr/bin/python

import copy
import itertools
import time



class CSP:
    def __init__(self):
        
        # Number of times backtrack is called
        self.counter = 0 
        # Number of times backtrack fails
        self.fails = 0

        self.variables = []

        # self.domains[i] is a list of legal values for variable i
        self.domains = {}

        # self.constraints[i][j] is a list of legal value pairs for
        # the variable pair (i, j)
        self.constraints = {}

    def add_variable(self, name, domain):
        """Add a new variable to the CSP. 'name' is the variable name
        and 'domain' is a list of the legal values for the variable.
        """
        self.variables.append(name)
        self.domains[name] = list(domain)
        self.constraints[name] = {}

    def get_all_possible_pairs(self, a, b):
        """Get a list of all possible pairs (as tuples) of the values in
        the lists 'a' and 'b', where the first component comes from list
        'a' and the second component comes from list 'b'.
        """
        return itertools.product(a, b)

    def get_all_arcs(self):
        """Get a list of all arcs/constraints that have been defined in
        the CSP. The arcs/constraints are represented as tuples (i, j),
        indicating a constraint between variable 'i' and 'j'.
        """
        return [ (i, j) for i in self.constraints for j in self.constraints[i] ]

    def get_all_neighboring_arcs(self, var):
        """Get a list of all arcs/constraints going to/from variable
        'var'. The arcs/constraints are represented as in get_all_arcs().
        """
        return [ (i, var) for i in self.constraints[var] ]

    def add_constraint_one_way(self, i, j, filter_function):
        """Add a new constraint between variables 'i' and 'j'. The legal
        values are specified by supplying a function 'filter_function',
        that returns True for legal value pairs and False for illegal
        value pairs. This function only adds the constraint one way,
        from i -> j. You must ensure that the function also gets called
        to add the constraint the other way, j -> i, as all constraints
        are supposed to be two-way connections!
        """
        if not j in self.constraints[i]:
            # First, get a list of all possible pairs of values between variables i and j
            self.constraints[i][j] = self.get_all_possible_pairs(self.domains[i], self.domains[j])

        # Next, filter this list of value pairs through the function
        # 'filter_function', so that only the legal value pairs remain
        self.constraints[i][j] = filter(lambda value_pair: filter_function(*value_pair), self.constraints[i][j])

    def add_all_different_constraint(self, variables):
        """Add an Alldiff constraint between all of the variables in the
        list 'variables'.
        """
        for (i, j) in self.get_all_possible_pairs(variables, variables):
            if i != j:
                self.add_constraint_one_way(i, j, lambda x, y: x != y)

    def backtracking_search(self):
        """This functions starts the CSP solver and returns the found
        solution.
        """
        # Make a so-called "deep copy" of the dictionary containing the
        # domains of the CSP variables. The deep copy is required to
        # ensure that any changes made to 'assignment' does not have any
        # side effects elsewhere.
        assignment = copy.deepcopy(self.domains)

        # Run AC-3 on all constraints in the CSP, to weed out all of the
        # values that are not arc-consistent to begin with
        self.inference(assignment, self.get_all_arcs())
        # Call backtrack with the partial assignment 'assignment'

        return self.backtrack(assignment)

    def backtrack(self, assignment):
        """The function 'Backtrack' from the pseudocode in the
        textbook.

        The function is called recursively, with a partial assignment of
        values 'assignment'. 'assignment' is a dictionary that contains
        a list of all legal values for the variables that have *not* yet
        been decided, and a list of only a single value for the
        variables that *have* been decided.

        When all of the variables in 'assignment' have lists of length
        one, i.e. when all variables have been assigned a value, the
        function should return 'assignment'. Otherwise, the search
        should continue. When the function 'inference' is called to run
        the AC-3 algorithm, the lists of legal values in 'assignment'
        should get reduced as AC-3 discovers illegal values.

        IMPORTANT: For every iteration of the for-loop in the
        pseudocode, you need to make a deep copy of 'assignment' into a
        new variable before changing it. Every iteration of the for-loop
        should have a clean slate and not see any traces of the old
        assignments and inferences that took place in previous
        iterations of the loop.
        """
        # As stated above, if all variables in assignment is 1
        # then all values have been set and we return assignment 
        if all(len(l) == 1 for l in assignment.values()):
            return assignment

        # Pick the next unnassigned variable that we are going to check     
        key, values = self.select_unassigned_variable(assignment)
        # Loop through all the allowed values of this square in the sudoku board
        for value in values:
            # Do a deepcopy cuz otherwise R.I.P
            deep = copy.deepcopy(assignment)
            # Checks if this current value is consistent with the rest
            # of the sudoku board 
            if self.check_consistency(deep, key, value):
                # IF it is consistent then we set this square to have this value 
                deep[key] = [value]
                # Do inference check for hyper optimized code
                if self.inference(deep, self.get_all_arcs()):
                    self.counter += 1
                    result = self.backtrack(deep)
                    if result is not False:
                        return result
                    else:
                        self.fails += 1
            else:
                # Continue looping through the values of the currently selected 
                # sudoku-square if the value was inconsistent with the board 
                continue
        return False



    def check_consistency(self, assignment, var, value):
        '''
        This method checks if the current value picked in self.backtrack()
        is consistent with the rest of the board
        '''
        consistent = True
        # Get all the constraint for this current square
        key_constraints = self.constraints[var]
        # Loop trough all the other squares and their values 
        for key, values in key_constraints.iteritems():
            # Create a list that makes combinations of this value and the 
            # other squared possible values 
            combinations = [(value, y) for y in assignment[key]]
            # Make a list of all the constraint that is between the two squares
            tmp = [x for x in combinations if x in values]
            # If the list is empty they values are not possible 
            if not tmp:
                return False
        return consistent

    def select_unassigned_variable(self, assignment):
        """The function 'Select-Unassigned-Variable' from the pseudocode
        in the textbook. Should return the name of one of the variables
        in 'assignment' that have not yet been decided, i.e. whose list
        of legal values has a length greater than one.
        """
        # Simply just pick the next value that has more than one value
        # in the variable list
        for key, value in assignment.iteritems():
            if len(value) > 1:
                return key, value

    def inference(self, assignment, queue):
        """The function 'AC-3' from the pseudocode in the textbook.
        'assignment' is the current partial assignment, that contains
        the lists of legal values for each undecided variable. 'queue'
        is the initial queue of arcs that should be visited.
        """
        # Do this as long as there is elements in the queue
        # e.g there is still more arcs to check 
        while queue:
            # Pop the first element in the queue
            xi, xj = queue.pop(0)
            # Do the revise check 
            if self.revise(assignment, xi, xj):
                # IF zero, CSP has no consistent soluton and AC-3 returns failure 
                if len(assignment[xi]) == 0:
                    return False
                # If NOT ZERO loop throuh the neighboring arcs of node
                # and append the neighbor and this node to the queue for further checking.
                # We do this so that we keep checking after we do changes and make sure 
                # all is gucci gang
                for n in self.get_all_neighboring_arcs(xi):
                    if n[0] != xj:
                        queue.append((n[0], xi))
        return True


    def revise(self, assignment, i, j):
        """The function 'Revise' from the pseudocode in the textbook.
        'assignment' is the current partial assignment, that contains
        the lists of legal values for each undecided variable. 'i' and
        'j' specifies the arc that should be visited. If a value is
        found in variable i's domain that doesn't satisfy the constraint
        between i and j, the value should be deleted from i's list of
        legal values in 'assignment'.
        """
        revised = False
        # For all the values in i's variables
        for x in assignment[i]:
            # if there exist NO possible values in the constraints between i and j
            # then remove this value from i
            if not any([(x,y) for y in assignment[j] if (x,y) in self.constraints[i][j]]):
                assignment[i].remove(x)
                revised = True
        return revised
            
         
def create_map_coloring_csp():
    """Instantiate a CSP representing the map coloring problem from the
    textbook. This can be useful for testing your CSP solver as you
    develop your code.
    """
    csp = CSP()
    states = [ 'WA', 'NT', 'Q', 'NSW', 'V', 'SA', 'T' ]
    edges = { 'SA': [ 'WA', 'NT', 'Q', 'NSW', 'V' ], 'NT': [ 'WA', 'Q' ], 'NSW': [ 'Q', 'V' ] }
    colors = [ 'red', 'green', 'blue' ]
    for state in states:
        csp.add_variable(state, colors)
    for state, other_states in edges.items():
        for other_state in other_states:
            csp.add_constraint_one_way(state, other_state, lambda i, j: i != j)
            csp.add_constraint_one_way(other_state, state, lambda i, j: i != j)
    return csp

def create_sudoku_csp(filename):
    """Instantiate a CSP representing the Sudoku board found in the text
    file named 'filename' in the current directory.
    """
    csp = CSP()
    board = map(lambda x: x.strip(), open(filename, 'r'))

    for row in range(9):
        for col in range(9):
            if board[row][col] == '0':
                csp.add_variable('%d-%d' % (row, col), map(str, range(1, 10)))
            else:
                csp.add_variable('%d-%d' % (row, col), [ board[row][col] ])

    for row in range(9):
        csp.add_all_different_constraint([ '%d-%d' % (row, col) for col in range(9) ])
    for col in range(9):
        csp.add_all_different_constraint([ '%d-%d' % (row, col) for row in range(9) ])
    for box_row in range(3):
        for box_col in range(3):
            cells = []
            for row in range(box_row * 3, (box_row + 1) * 3):
                for col in range(box_col * 3, (box_col + 1) * 3):
                    cells.append('%d-%d' % (row, col))
            csp.add_all_different_constraint(cells)

    return csp

def print_sudoku_solution(solution):
    """Convert the representation of a Sudoku solution as returned from
    the method CSP.backtracking_search(), into a human readable
    representation.
    """
    for row in range(9):
        for col in range(9):
            print solution['%d-%d' % (row, col)][0],
            if col == 2 or col == 5:
                print '|',
        print
        if row == 2 or row == 5:
            print '------+-------+------'


if __name__ == "__main__":
    

    sudoku_easy = "boards\\easy.txt"
    sudoku_medium = "boards\\medium.txt"
    sudoku_hard = "boards\\hard.txt"
    sudoku_veryhard = "boards\\veryhard.txt"

    csp = create_sudoku_csp(sudoku_easy)
    print "Running sudoku level: EASY" 
    t0 = time.time()
    x = csp.backtracking_search()
    t1 = time.time()
    print_sudoku_solution(x)
    print "\nTime:", t1-t0
    print "Number of times backtrack() was called: ", csp.counter
    print "Number of times backtrack() failes: ", csp.fails

  
