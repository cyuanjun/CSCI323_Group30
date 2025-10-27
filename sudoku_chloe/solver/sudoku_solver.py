
import numpy as np

def find_empty(grid):
    for y in range(9):
        for x in range(9):
            if grid[y][x] == 0:
                return (y, x)
    return None


def is_valid(grid, num, pos):
    row, col = pos

    # Check row
    if num in grid[row]:
        return False

    # Check column
    if num in grid[:, col]:
        return False

    # Check 3x3 box
    box_x = col // 3
    box_y = row // 3

    for i in range(box_y * 3, box_y * 3 + 3):
        for j in range(box_x * 3, box_x * 3 + 3):
            if grid[i][j] == num:
                return False

    return True


def solve_sudoku(grid):
    empty = find_empty(grid)
    if not empty:
        return True  # Puzzle solved
    else:
        row, col = empty

    for num in range(1, 10):
        if is_valid(grid, num, (row, col)):
            grid[row][col] = num

            if solve_sudoku(grid):
                return True

            grid[row][col] = 0  # Reset (backtrack)

    return False


def print_grid(grid):
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("-" * 21)
        for j in range(9):
            if j % 3 == 0 and j != 0:
                print("|", end=" ")
            print(grid[i][j], end=" ")
        print()


if __name__ == "__main__":
    sample_grid = np.array([
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ])

    print("Initial Sudoku:")
    print_grid(sample_grid)

    if solve_sudoku(sample_grid):
        print("\nSolved Sudoku:")
        print_grid(sample_grid)
    else:
        print("\nNo solution found.")
