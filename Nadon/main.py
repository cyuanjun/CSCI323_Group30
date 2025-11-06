
import cv2
import numpy as np
from imageRecognition.preprocess import preprocess_image
from imageRecognition.digit_recognition import recognize_digits
from solver.sudoku_solver import solve_sudoku

def display_solution(original_img, grid, solved_grid):
    side = 450
    cell_size = side // 9
    display = original_img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for y in range(9):
        for x in range(9):
            if grid[y][x] == 0:  
                num = solved_grid[y][x]
                x_pos = int(x * cell_size + cell_size / 3)
                y_pos = int(y * cell_size + cell_size / 1.4)
                cv2.putText(display, str(num), (x_pos, y_pos),
                            font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Solved Sudoku", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    print("Loading and processing image...")
    image_path = "/Users/nadonpanwong/Desktop/CSCI323_Group30/Nadon/testImages/image6.png"

    warped_board, cells = preprocess_image(image_path)

    print("Recognizing digits...")
    grid = recognize_digits(cells)
    print("Detected Sudoku Grid:")
    print(grid)

    print("Solving Sudoku...")
    solved_grid = grid.copy()
    if solve_sudoku(solved_grid):
        print("Sudoku Solved Successfully!")
        print(solved_grid)
    else:
        print("No valid solution found.")

    display_solution(warped_board, grid, solved_grid)


if __name__ == "__main__":
    main()

