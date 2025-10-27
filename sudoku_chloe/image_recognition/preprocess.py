import cv2
import numpy as np


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    board_contour = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            board_contour = approx
            break

    if board_contour is None:
        raise ValueError("❌ Sudoku board not found in the image.")

    def reorder_points(pts):
        pts = pts.reshape((4, 2))
        new_pts = np.zeros((4, 2), dtype=np.float32)
        add = pts.sum(1)
        diff = np.diff(pts, axis=1)

        new_pts[0] = pts[np.argmin(add)]     # top-left
        new_pts[1] = pts[np.argmin(diff)]    # top-right
        new_pts[2] = pts[np.argmax(diff)]    # bottom-left
        new_pts[3] = pts[np.argmax(add)]     # bottom-right
        return new_pts

    pts1 = reorder_points(board_contour)
    side = 450 
    pts2 = np.float32([[0, 0], [side, 0], [0, side], [side, side]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(original, matrix, (side, side))

    cells = []
    cell_size = side // 9
    for y in range(9):
        row = []
        for x in range(9):
            x_start = x * cell_size
            y_start = y * cell_size
            cell = warped[y_start:y_start + cell_size, x_start:x_start + cell_size]
            row.append(cell)
        cells.append(row)

    return warped, cells



def visualize_cells(warped_board):
    display = warped_board.copy()
    side = warped_board.shape[0]
    cell_size = side // 9

    font = cv2.FONT_HERSHEY_SIMPLEX
    index = 0

    for y in range(9):
        for x in range(9):
            x_center = int(x * cell_size + cell_size / 3)
            y_center = int(y * cell_size + cell_size / 1.8)

            cv2.putText(display, str(index), (x_center, y_center),
                        font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            index += 1
            
    for i in range(1, 9):
        cv2.line(display, (0, i * cell_size), (side, i * cell_size), (200, 200, 200), 1)
        cv2.line(display, (i * cell_size, 0), (i * cell_size, side), (200, 200, 200), 1)

    cv2.imshow("Cell Visualization", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    warped_board, cells = preprocess_image("test_images/image1.png")
    print(f"✅ Board detected successfully! Grid size: {len(cells)}x{len(cells[0])}")
    visualize_cells(warped_board)

