import cv2
import numpy as np
import os

fonts = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_PLAIN,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
]

os.makedirs("sudoku_digits", exist_ok=True)

for digit in range(1, 10):
    digit_dir = f"sudoku_digits/{digit}"
    os.makedirs(digit_dir, exist_ok=True)

    for i in range(400):  
        img = np.zeros((28, 28), dtype=np.uint8)

        font = np.random.choice(fonts)
        scale = np.random.uniform(0.7, 1.4)
        thickness = np.random.randint(1, 3)

        text = str(digit)
        size = cv2.getTextSize(text, font, scale, thickness)[0]
        x = (28 - size[0]) // 2 + np.random.randint(-2, 2)
        y = (28 + size[1]) // 2 + np.random.randint(-2, 2)

        cv2.putText(img, text, (x, y), font, scale, (255), thickness, cv2.LINE_AA)

        cv2.imwrite(f"{digit_dir}/{digit}_{i}.png", img)
