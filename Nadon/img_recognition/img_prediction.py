import cv2
import numpy as np
import matplotlib.pyplot as plt

# ...existing code...
# helper to open a camera on macOS (use AVFoundation backend)
# def open_camera(preferred_index=None, max_try=5):
#     backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
#     indices = list(range(max_try))
#     if preferred_index is not None:
#         # try preferred first
#         indices = [preferred_index] + [i for i in indices if i != preferred_index]
#     for backend in backends:
#         for idx in indices:
#             cap = cv2.VideoCapture(idx, backend)
#             if not cap.isOpened():
#                 cap.release()
#                 continue
#             # try to grab a frame to confirm this device works
#             ret, _ = cap.read()
#             if ret:
#                 print(f"Opened camera index {idx} using backend {backend}")
#                 return cap, idx, backend
#             cap.release()
#     return None, None, None

# # Try to open the built-in webcam first (try index 0 then others)
# cap, used_idx, used_backend = open_camera(preferred_index=0, max_try=6)
# if cap is None:
#     print("ERROR: Could not open any camera. Check macOS Camera permissions (System Settings → Privacy & Security → Camera) and make sure iPhone Continuity Camera is not forcing selection.")
#     exit(1)

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# original code
def getContours(img, original_img):
    
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        
        area = cv2.contourArea(cnt)
        if area > 30000:
            cv2.drawContours(original_img, cnt, -1, (0, 255, 0), 2)
            
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if approx.size < 8:
                # skip if not enough points
                continue
            ax = approx.item(0)
            ay = approx.item(1)
            bx = approx.item(2)
            by = approx.item(3)
            cx = approx.item(4)
            cy = approx.item(5)
            dx = approx.item(6)
            dy = approx.item(7)

            width,height = 900,900
            
            pts1 = np.float32([[bx,by],[ax,ay],[cx,cy],[dx,dy]])
            pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])

            matrix = cv2.getPerspectiveTransform(pts1,pts2)
            img_perspective = cv2.warpPerspective(original_img, matrix, (width, height))
            img_corners = cv2.cvtColor(img_perspective, cv2.COLOR_BGR2GRAY)
            
            for x in range(0 , 900):
                for y in range(0 , 900):
                    if img_corners[x , y] < 200:
                        img_corners[x , y] = 0
                    else:
                        img_corners[x , y] = 255

            cv2.imshow("Corners", img_corners)
            
            # cell = img_corners[0:120 , 0:120]
            
            
            # plt.imshow(cell, cmap="gray")
            # plt.show()
            
            crop_value = 20
            
            for y in range(1, 9):
                for x in range(1, 9):
                    plt.imshow(img_corners[y*100-100+crop_value:y*100-crop_value , x*100-100+crop_value:x*100-crop_value], cmap="gray")
                    plt.show()
            
while True:
    success, img = cap.read()
    # Turning the original image to canny
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 3)
    imgCanny = cv2.Canny(imgBlur, 50, 50)
    img_copy = img.copy()
    
    # Getting contours
    getContours(imgCanny, img_copy)
    
    cv2.imshow("Webcam", img_copy)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

# cap.release()