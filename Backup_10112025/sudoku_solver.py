#######################################################################################################################
# Evaluation
stats = {"calls": 0, "backtracks": 0}

def reset_stats():
    stats["calls"] = 0
    stats["backtracks"] = 0

def get_stats():
    # return a copy to avoid accidental mutation
    return {"calls": stats["calls"], "backtracks": stats["backtracks"]}

def print_board(board, title="Sudoku Board"):
    print(f"\n=== {title} ===")
    for r in range(9):
        if r % 3 == 0 and r != 0:
            print("-" * 21)
        for c in range(9):
            if c % 3 == 0 and c != 0:
                print("|", end=" ")
            print(board[r][c], end=" ")
        print()

#######################################################################################################################
# BackTracking

def find_empty(board):
    """
    Find the next empty cell on (value 0) on the Sudoku.
    Scans the board row by row, and column by column
    :param board: Sudoku board
    """
    for r in range(9):  # search row 1-9
        for c in range(9):  # search column 1-9
            if board[r][c] == 0:  # once it found empty box which value is 0
                return r, c  # return location of box
    return None

def is_valid(board, r, c, v):
    """
    Check whether placing value v at position (r, c) is valid
    * v is not in the same row
    * v is not in the same column
    * v is not in the same 3 x 3 subgrid
    :param board: Sudoku board
    :param r: row
    :param c: column
    :param v: value
    """
    for i in range(9):  # loop thru each column
        if board[r][i] == v:  # check whether same number exists
            return False

    for i in range(9):  # loop thru each row
        if board[i][c] == v:  # check whether same number exists
            return False

    gb_v = (r // 3) * 3  # find grid box 3 x 3
    gb_h = (c // 3) * 3

    for rr in range(gb_v, gb_v+ 3):  # loop thru the column of grid box
        for cc in range(gb_h, gb_h+3):  # loop thru the row of grid box
            if board[rr][cc] == v:  # check whether same number exists
                return False
    return True

def sudoku_solve(board):
    stats["calls"] += 1  # added to evaluate recursive call
    empty = find_empty(board)  # execute find_empty method
    if not empty:  # there's no empty box anymore return true
        return True
    r, c = empty  # declare (r, c) is empty

    for v in range(1, 10):  # the possible number can put into the empty box
        if is_valid(board, r, c, v):  # Execute is_valid method, which will check whether there's same number with v exsts
            board[r][c] = v  # assign box(r,c) as v
            if sudoku_solve(board):  # propagate
                return True
            board[r][c] = 0  # backtrack
            stats["backtracks"] += 1  # count the number of backtrack
    return False

