import cv2
import numpy as np
import pytesseract

def extract_digit(cell_img):

    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    kernel = np.ones((2, 2), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(clean)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 30:  
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)

    digit_only = cv2.bitwise_and(clean, mask)

    white_ratio = np.sum(digit_only > 0) / (digit_only.shape[0] * digit_only.shape[1])
    if white_ratio < 0.02:
        return 0

    config = '--psm 10 --oem 3 -c tessedit_char_whitelist=123456789'
    text = pytesseract.image_to_string(digit_only, config=config).strip()

    if text.isdigit():
        return int(text)
    return 0


def recognize_digits(cells):
    sudoku_grid = np.zeros((9, 9), dtype=int)

    for y in range(9):
        for x in range(9):
            digit = extract_digit(cells[y][x])
            sudoku_grid[y, x] = digit

    return sudoku_grid


if __name__ == "__main__":
    from image_recognition.preprocess import preprocess_image

    warped_board, cells = preprocess_image("test_images/image2.png")
    grid = recognize_digits(cells)

    print("✅ detected sudoku grid:")
    print(grid)
