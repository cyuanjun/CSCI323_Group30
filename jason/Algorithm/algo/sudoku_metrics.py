# sudoku_metrics.py
import time

def is_solved_correctly(board):
    nums = set(range(1, 10))
    # rows
    for row in board:
        if set(row) != nums:
            return False
    # cols
    for c in range(9):
        if set(board[r][c] for r in range(9)) != nums:
            return False
    # boxes
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            box = [board[r][c] for r in range(br, br+3) for c in range(bc, bc+3)]
            if set(box) != nums:
                return False
    return True

def benchmark_solver(solver_func, board_loader, reset_stats, get_stats):
    """
    Run solver and return metrics dict:
    - solved (bool), accuracy (0/100), time_sec (float), recursive_calls (int), backtracks (int)
    """
    reset_stats()
    board = board_loader()

    start = time.perf_counter()
    solved = solver_func(board)
    end = time.perf_counter()

    elapsed = end - start
    acc_ok = is_solved_correctly(board) if solved else False
    s = get_stats()

    return {
        "solved": solved,
        "accuracy": 100 if acc_ok else 0,
        "time_sec": elapsed,
        "recursive_calls": s["calls"],
        "backtracks": s["backtracks"],
    }
