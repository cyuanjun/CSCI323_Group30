import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

def pickImages(grid,Img):
    
    DATADROP = '/Users/nadonpanwong/Desktop/CSCI323_Group30/Nadon/IMG/'
    crop_value = 20
    
    listNum = []
    for i in range(0,9):
        for j in range(0,9):
            if grid[i][j] not in listNum:
                print(grid[i][j])
                listNum.append(grid[i][j])
                J = j+1
                I = i+1
                cell = Img[I*100-100+crop_value:I*100-crop_value , J*100-100+crop_value:J*100-crop_value]

                cv2.imwrite(DATADROP + str(grid[i][j]) + '/img{}.png'.format((i+1)*(j+1)), cell)

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
            
            pickImages(grid266 , img_corners)
            
            # cell = img_corners[0:120 , 0:120]
            
            
            # plt.imshow(cell, cmap="gray")
            # plt.show()
            
            # crop_value = 20
            
            # for y in range(1, 9):
            #     for x in range(1, 9):
            #         plt.imshow(img_corners[y*100-100+crop_value:y*100-crop_value , x*100-100+crop_value:x*100-crop_value], cmap="gray")
            #         plt.show()
            
while True:
    success, img = cap.read()
    # Turning the original image to canny
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 3)
    imgCanny = cv2.Canny(imgBlur, 50, 50)
    img_copy = img.copy()
    
    grid266 = [[3, 0, 0, 8, 0, 1, 0, 0, 2],
               [2, 0, 1, 0, 3, 0, 6, 0, 4],
               [0, 0, 0, 2, 0, 4, 0, 0, 0],
               [8, 0, 9, 0, 0, 0, 1, 0, 6],
               [0, 6, 0, 0, 0, 0, 0, 5, 0],
               [7, 0, 2, 0, 0, 0, 4, 0, 9],
               [0, 0, 0, 5, 0, 9, 0, 0, 0],
               [9, 0, 4, 0, 8, 0, 7, 0, 5],
               [6, 0, 0, 1, 0, 7, 0, 0, 3]]
    
    # Getting contours
    getContours(imgCanny, img_copy)
    
    cv2.imshow("Webcam", img_copy)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break