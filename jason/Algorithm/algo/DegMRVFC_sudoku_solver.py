#######################################################################################################################
# Evaluation
stats = {"calls": 0, "backtracks": 0}

def reset_stats():
    stats["calls"] = 0
    stats["backtracks"] = 0

def get_stats():
    # return a copy to avoid accidental mutation
    return {"calls": stats["calls"], "backtracks": stats["backtracks"]}

#######################################################################################################################
# BackTracking

def find_empty(board):  # find empty box from the first row
    for r in range(9):  # search row 1-9
        for c in range(9):  # search column 1-9
            if board[r][c] == 0:  # once it found empty box which value is 0
                return r, c  # return location of box
    return None

def is_valid(board, r, c, v):
    for i in range(9):  # loop thru each column
        if board[r][i] == v:  # check whether same number exists
            return False

    for i in range(9):  # loop thru each row
        if board[i][c] == v:  # check whether same number exists
            return False

    gb_v = (r // 3) * 3  # find grid box 3 x 3
    gb_h = (c // 3) * 3

    for rr in range(gb_v, gb_v+ 3):  # loop thru the column of grid box 3x3
        for cc in range(gb_h, gb_h+3):  # loop thru the row of grid box 3x3
            if board[rr][cc] == v:  # check whether same number exists
                return False
    return True

def get_candidates(board, r, c):
    """Return list of all valid numbers that can go into (r, c)"""
    candidates = []
    for v in range (1, 10):
        if is_valid(board, r, c, v):
            candidates.append(v)
    return candidates

def find_mrv(board):
    """
    Find the empty cell with the minimum remaining values
    returns r, c, best_candidates
    """
    best_cell = None
    best_candidates = None
    min_len = 10
    max_degree = -1

    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                candidates = get_candidates(board, r, c)

                Length = len(candidates)

                if Length == 0:
                    # dead end
                    continue

                if Length < min_len:
                    # found lesser candidate
                    min_len = Length
                    best_cell = (r, c)
                    best_candidates = candidates
                    max_degree = count_empty_neighbors(board, r, c)

                elif Length == min_len:
                    # if the number of candidates is same, compare with degree
                    degree = count_empty_neighbors(board, r, c)
                    if degree > max_degree:
                        best_cell = (r, c)
                        best_candidates = candidates
                        max_degree = degree


        if min_len == 1:
            break

    if best_cell is None:
        return None, None, None

    r,c = best_cell
    return r, c, best_candidates

def forward_checking(board):
    """
    Check if every empty cell still has at least one valid candidate.
    If some empty cell has no possible value, return False (dead end).
    """
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                has_candidate = False
                for v in range(1, 10):
                    if is_valid(board, r, c, v):
                        has_candidate = True
                        break

                if not has_candidate:
                    return False
    return True

def count_empty_neighbors(board, r , c):
    """ The number of empty neighbors that is in the same row and column with (r, c)"""
    count = 0
    # count the number of empty box in the same row
    for column in range(9):
        if column != c and board[r][column] == 0:
            count += 1

    # count the number of empty box in the same column
    for row in range(9):
        if row != r and board[row][c] == 0:
            count += 1

    # Within the 3x3 matrix
    gb_v = (r // 3) * 3
    gb_h = (c // 3) * 3
    for row in range(gb_v, gb_v+3):
        for column in range(gb_h, gb_h+3):
            if (row != r or column != c) and board[row][column] == 0:
                count += 1

    return count


def sudoku_solve(board):
    stats["calls"] += 1  # added to evaluate recursive call

    r, c, candidates = find_mrv(board)

    # No empty cell : solved
    if r is None:
        return True

    for v in candidates:  # the possible number can put into the empty box
        if is_valid(board, r, c, v):  # Execute is_valid method, which will check whether there's same number with v exsts
            board[r][c] = v  # assign box(r,c) as v

            if forward_checking(board):
                if sudoku_solve(board):  # propagate
                    return True

            board[r][c] = 0  # backtrack
            stats["backtracks"] += 1  # count the number of backtrack
    return False
