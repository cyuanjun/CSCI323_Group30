import sudoku_solver as bt
import MRV_sudoku_solver as mg
import MRVFC_sudoku_solver as fc
import DegMRVFC_sudoku_solver as dfc
import lcvDegMrvFc_sudoku_solver as ldmfc
import FINAL_digit_recognition as dr
import numpy as np
import cv2
from sudoku_metrics import benchmark_solver



def main(img_path):

    arr, warped_image = dr.digit_reg(img_path)

    def load_board():
        return arr.tolist()

    res_bt = benchmark_solver(bt.sudoku_solve, load_board, bt.reset_stats, bt.get_stats)

    res_mg = benchmark_solver(mg.sudoku_solve, load_board, mg.reset_stats, mg.get_stats)

    res_fc = benchmark_solver(fc.sudoku_solve, load_board, fc.reset_stats, fc.get_stats)

    res_dfc = benchmark_solver(dfc.sudoku_solve, load_board, dfc.reset_stats, dfc.get_stats)

    res_ldmfc = benchmark_solver(ldmfc.sudoku_solve, load_board, ldmfc.reset_stats, ldmfc.get_stats)

    # === Basic Backtracking ===
    print("=== Benchmark Results: Basic Backtracking ===")
    for k, v in res_bt.items():
        print(f"{k:20s}: {v}")
    board_bt = load_board()
    bt.sudoku_solve(board_bt)
    bt.print_board(board_bt, "Basic Backtracking Solved Board")

    # === MRV ===
    print("\n=== Benchmark Results: MRV Backtracking ===")
    for k, v in res_mg.items():
        print(f"{k:20s}: {v}")
    board_mg = load_board()
    mg.sudoku_solve(board_mg)
    mg.print_board(board_mg, "MRV Solved Board")

    # === MRV + Forward Checking ===
    print("\n=== Benchmark Results: MRV + Forward Checking ===")
    for k, v in res_fc.items():
        print(f"{k:20s}: {v}")
    board_fc = load_board()
    fc.sudoku_solve(board_fc)
    fc.print_board(board_fc, "MRV + Forward Checking Solved Board")

    # === MRV + FC + Degree ===
    print("\n=== Benchmark Results: MRV + FC + Degree ===")
    for k, v in res_dfc.items():
        print(f"{k:20s}: {v}")
    board_dfc = load_board()
    dfc.sudoku_solve(board_dfc)
    dfc.print_board(board_dfc, "MRV + FC + Degree Solved Board")

    # === MRV + FC + Degree + LCV ===
    print("\n=== Benchmark Results: MRV + FC + Degree + LCV ===")
    for k, v in res_ldmfc.items():
        print(f"{k:20s}: {v}")
    board_ldmfc = load_board()
    ldmfc.sudoku_solve(board_ldmfc)
    ldmfc.print_board(board_ldmfc, "MRV + FC + Degree + LCV Solved Board")

    print("=" * 70)
    solved_board_bt = np.array(board_bt)
    solved_board_mg = np.array(board_mg)
    solved_board_fc = np.array(board_fc)
    solved_board_dfc = np.array(board_dfc)
    solved_board_ldmfc = np.array(board_ldmfc)

    blank = dr.draw_board_on_image(warped_image)
    sol_bt = dr.draw_board_on_image(warped_image, arr, solved_board_bt)
    sol_mg = dr.draw_board_on_image(warped_image, arr, solved_board_mg)
    sol_fc = dr.draw_board_on_image(warped_image, arr, solved_board_fc)
    sol_dfc = dr.draw_board_on_image(warped_image, arr, solved_board_dfc)
    sol_ldmfc = dr.draw_board_on_image(warped_image, arr, solved_board_ldmfc)

    images = [blank, sol_bt, sol_mg, sol_fc, sol_dfc, sol_ldmfc]
    labels = [
        "Blank",
        "Basic Backtracking",
        "MRV",
        "MRV + FC",
        "MRV + FC + Degree",
        "MRV + FC + Degree + LCV"
    ]

    show_solver_row(images, labels)



def show_solver_row(images, labels, window_title="Solver Results"):

    assert len(images) == len(labels) and len(images) > 0, "images and labels must match and be non-empty"

    def to3(x):
        return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) if (x.ndim == 2) else x

    def label(im, text, color=(0,255,0)):

        h, w = im.shape[:2]
        label_h = 50
        strip = np.full((label_h, w, 3), (0,0,0), np.uint8)
        cv2.putText(strip, text, (10, int(label_h*0.75)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        return np.vstack((im, strip))


    tiles = [
        label(cv2.resize(to3(img), (450,450)), lab)
        for img, lab in zip(images, labels)
    ]

    vis = np.hstack(tiles)
    cv2.imshow(window_title, vis)
    # cv2.imwrite("solver_row.png", vis)  -------------------------------------- Uncomment if u want to print out the output
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return vis



if __name__ == "__main__":

    while True:

        print("=" * 70)
        img_path = input("Enter path of sudoku image to be solved or (N) to end: ")

        if img_path.upper() == "N":
            break
        
        img = cv2.imread(img_path)

        if img is None:
            print("-" * 70)
            print("Error: File not found, please check image path!")
            print("(Remember to add file extension (.png / .jpg!))")

            continue

        main(img_path)








