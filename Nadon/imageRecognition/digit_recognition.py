import cv2
import numpy as np
from tensorflow import keras
import os

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'cnnModel', 'models', 'sudoku_digit_model.h5')
MODEL = keras.models.load_model(MODEL_PATH)
print(f"Loaded model from {MODEL_PATH}")

def extract_digit(cell_img):
    """Extract and recognize digit using CNN model"""
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    
    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    if cv2.countNonZero(thresh) < 40:
        return 0
    
    confidence_threshold = 0.5 if digit in [1,7] else 0.7
    if confidence > confidence_threshold:
            return digit

    # Remove noise
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find largest contour (the digit)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0
    
    # Filter small contours (noise)
    contours = [c for c in contours if cv2.contourArea(c) > 50]
    if not contours:
        return 0
    
    # Get bounding box of largest contour
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    
    # Check if contour is too small (likely empty cell)
    if w < 5 or h < 5:
        return 0
    
    # Extract and center the digit
    digit_roi = thresh[y:y+h, x:x+w]
    
    # Resize to 28x28 with padding
    digit_size = max(w, h)
    square = np.zeros((digit_size, digit_size), dtype=np.uint8)
    x_offset = (digit_size - w) // 2
    y_offset = (digit_size - h) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = digit_roi
    
    # Resize to 28x28 for model
    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Normalize
    normalized = resized.astype('float32') / 255.0
    normalized = normalized.reshape(1, 28, 28, 1)
    
    # Predict
    prediction = MODEL.predict(normalized, verbose=0)
    digit = np.argmax(prediction)
    confidence = np.max(prediction)
    
    # Only return if confident and not 0
    if confidence > 0.7 and digit != 0:
        return digit
    
    return 0

def recognize_digits(cells):
    sudoku_grid = np.zeros((9, 9), dtype=int)
    
    for y in range(9):
        for x in range(9):
            digit = extract_digit(cells[y][x])
            sudoku_grid[y, x] = digit
    
    return sudoku_grid