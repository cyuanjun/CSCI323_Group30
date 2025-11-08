import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "digit_model_sudoku.h5")
MODEL_PATH = os.path.abspath(MODEL_PATH)

print("📌 Loading model from:", MODEL_PATH) 
model = load_model(MODEL_PATH)

def normalize_digit(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x,y,w,h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    digit = img[y:y+h, x:x+w]

    digit = cv2.resize(digit, (20, 20))

    padded = np.zeros((28, 28), dtype=np.uint8)
    padded[4:24, 4:24] = digit

    return padded


def extract_digit(cell_img):
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        17, 5
    )

    thresh = cv2.dilate(thresh, np.ones((2,2), np.uint8), iterations=1)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8))

    h, w = thresh.shape
    thresh = thresh[int(h*0.15):int(h*0.85), int(w*0.15):int(w*0.85)]

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0

    contours = [c for c in contours if cv2.contourArea(c) > 40]
    if not contours:
        return 0

    x,y,w,h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    digit = thresh[y:y+h, x:x+w]

    digit = normalize_digit(digit)
    if digit is None:
        return 0

    digit = digit.astype("float32") / 255.0
    digit = digit.reshape(1, 28, 28, 1)

    pred = model.predict(digit, verbose=0)
    num = np.argmax(pred)      
    conf = np.max(pred)        


    if conf < 0.50:
        return 0

    if num == 2 and conf < 0.75:  
        h, w = thresh.shape
        vertical_strength = np.sum(thresh[:, w//2-1:w//2+1] > 0) / (h * 2)

        if vertical_strength > 0.15:
            return 1

    return num + 1 



def recognize_digits(cells):
    sudoku_grid = np.zeros((9, 9), dtype=int)

    for y in range(9):
        for x in range(9):
            digit = extract_digit(cells[y][x])
            sudoku_grid[y, x] = digit

    return sudoku_grid
