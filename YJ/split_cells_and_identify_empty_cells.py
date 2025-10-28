import cv2
import numpy as np

# ---------- Configuration ----------
img_path = "Images/image1.png"
img_height, img_width = 450, 450
CELL_DISPLAY_SIZE = 50
CELL_SPACING = 5
EMPTY_FILL_RATIO = 0.05  # fraction of white pixels to consider a cell empty

# ---------- Functions ----------

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image at {img_path}")

    img = cv2.resize(img, (img_width, img_height))

    cv2.imshow("Input image (resized)", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_thresh = cv2.adaptiveThreshold(
        img_blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    img_thresh = cv2.bitwise_not(img_thresh)
    return img, img_thresh

def find_biggest_contour(img, img_thresh):
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest_area = 0
    biggest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if area > biggest_area and len(approx) == 4:
                biggest_area = area
                biggest_contour = approx
    if biggest_contour is None:
        raise Exception("No suitable Sudoku contour found!")

    points = biggest_contour.reshape((4, 2)).astype(np.float32)
    new_points = np.zeros((4, 2), dtype=np.float32)
    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]
    new_points[2] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[3] = points[np.argmax(diff)]
    return new_points

def warp_image(img, corners):
    destination = np.array([
        [0, 0],
        [img_width - 1, 0],
        [img_width - 1, img_height - 1],
        [0, img_height - 1]
    ], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(corners, destination)
    warped = cv2.warpPerspective(img, matrix, (img_width, img_height))
    return warped

def crop_cell_margin(cell, margin_ratio=0.05):
    h, w = cell.shape[:2]
    top = int(h * margin_ratio)
    bottom = int(h * (1 - margin_ratio))
    left = int(w * margin_ratio)
    right = int(w * (1 - margin_ratio))
    return cell[top:bottom, left:right]

def split_cells_to_matrix(warped_img, crop_margin=True):
    
    h, w = warped_img.shape[:2]
    cell_h, cell_w = h // 9, w // 9
    grid = []
    for row in range(9):
        row_cells = []
        for col in range(9):
            x0, y0 = col * cell_w, row * cell_h
            x1, y1 = (col + 1) * cell_w, (row + 1) * cell_h
            cell = warped_img[y0:y1, x0:x1]
            if crop_margin:
                cell = crop_cell_margin(cell, margin_ratio=0.2)
            row_cells.append(cell)
        grid.append(row_cells)
    return grid

def binarize_cell(cell):
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    bin_cell = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    return bin_cell

def is_cell_empty(cell, fill_ratio=EMPTY_FILL_RATIO):
    bin_cell = binarize_cell(cell)
    h, w = bin_cell.shape
    white_pixels = cv2.countNonZero(bin_cell)
    return (white_pixels / (h * w)) < fill_ratio

def print_sudoku_placeholder_with_separators(grid_cells):
    sudoku_grid = []
    for row in grid_cells:
        row_symbols = []
        for cell in row:
            if is_cell_empty(cell):
                row_symbols.append('0')  # empty cell
            else:
                row_symbols.append('X')  # filled placeholder
        sudoku_grid.append(row_symbols)

    print("Sudoku Grid (0 = empty, X = placeholder):\n")
    for i, row in enumerate(sudoku_grid):
        row_str = ""
        for j, val in enumerate(row):
            row_str += val + " "
            if (j + 1) % 3 == 0 and j < 8:
                row_str += "| "
        print(row_str.strip())
        if (i + 1) % 3 == 0 and i < 8:
            print("-" * 21)

def display_cells_grid(grid_cells, cell_size=CELL_DISPLAY_SIZE, spacing=CELL_SPACING):
    """Display all 81 cells in one window with spacing."""
    grid_img_height = 9 * cell_size + 8 * spacing
    grid_img_width = 9 * cell_size + 8 * spacing
    grid_img = np.ones((grid_img_height, grid_img_width), dtype=np.uint8) * 255  # white background

    for row in range(9):
        for col in range(9):
            cell = grid_cells[row][col]
            cell_gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            cell_resized = cv2.resize(cell_gray, (cell_size, cell_size))

            y_start = row * (cell_size + spacing)
            y_end = y_start + cell_size
            x_start = col * (cell_size + spacing)
            x_end = x_start + cell_size

            grid_img[y_start:y_end, x_start:x_end] = cell_resized

    cv2.imshow("Individual cropped cells", grid_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def display_cells_grid_with_placeholders(grid_cells, cell_size=CELL_DISPLAY_SIZE, spacing=CELL_SPACING):
    
    grid_img_height = 9 * cell_size + 8 * spacing
    grid_img_width = 9 * cell_size + 8 * spacing
    grid_img = np.ones((grid_img_height, grid_img_width), dtype=np.uint8) * 255  # white background

    for row in range(9):
        for col in range(9):
            cell = grid_cells[row][col]
            cell_gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            cell_resized = cv2.resize(cell_gray, (cell_size, cell_size))

            y_start = row * (cell_size + spacing)
            y_end = y_start + cell_size
            x_start = col * (cell_size + spacing)
            x_end = x_start + cell_size

            grid_img[y_start:y_end, x_start:x_end] = cell_resized

            # Overlay placeholder text
            if is_cell_empty(cell):
                text = '0'
            else:
                text = 'X'

            # Put text in the center of the cell
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = x_start + (cell_size - text_size[0]) // 2
            text_y = y_start + (cell_size + text_size[1]) // 2
            cv2.putText(grid_img, text, (text_x, text_y), font, font_scale, (0,), thickness, cv2.LINE_AA)

    cv2.imshow("Individual cropped cells with Placeholders", grid_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# ---------- Main Pipeline ----------
def main():
    img, img_thresh = preprocess_image(img_path)
    corners = find_biggest_contour(img, img_thresh)
    warped_img = warp_image(img, corners)
    sudoku_cells = split_cells_to_matrix(warped_img, crop_margin=True)
    print_sudoku_placeholder_with_separators(sudoku_cells)
    display_cells_grid(sudoku_cells)
    display_cells_grid_with_placeholders(sudoku_cells)


if __name__ == "__main__":
    main()
