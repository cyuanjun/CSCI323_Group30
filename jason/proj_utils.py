import cv2
import numpy as np

##### 1. Preprocessing Image
def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to gray scale
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # Add Gaussian blur to find out contour easier
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)
    # Appy adaptive threshold
    return imgThreshold

#### 3. Reorder
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))  # reshapes them into a simple (4,2) array
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)  # X + Y
    myPointsNew[0] = myPoints[np.argmin(add)]  # find smallest value among add --> top-left
    myPointsNew[3] = myPoints[np.argmax(add)]  # find biggest value among add --> bottom-right

    diff = np.diff(myPoints, axis=1)
    #  y-x
    myPointsNew[1] = myPoints[np.argmin(diff)]  # smallest y-x value --> top-right
    myPointsNew[2] = myPoints[np.argmax(diff)]  # biggest y-x value --> bottom-left

    return myPointsNew

####  3. Finding the biggest contour that is the sudoku puzzle
def biggestContour(contours):  # sending all the contours in this method
    biggest = np.array([])  # it will store the coordinates of the larges rectangular contour
    max_area = 0  # it keeps track of the largest contour area found so far
    for i in contours:
        area = cv2.contourArea(i)  # calulates the area (size in pixels) of the contour
        if area > 50:  # Only consider contours with area is greater than 50 to ignore small noise
            peri = cv2.arcLength(i, True)  # Calculate Perimeter
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)  # Finds corners
            if area > max_area and len(approx) == 4:  # len(approx) == 4 means the contour has 4 corners --> rectangle
                biggest = approx  # if its area is larger than any previous contour, becomes biggest
                max_area = area
                return biggest, max_area

#### 4 - To split the image into 81 different images
def splitBoxes(img):
    rows = np.vsplit(img, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            boxes.append(box)
    return boxes



#### 6 - TO STACK ALL THE IMAGES IN ONE WINDOW
def stackImages(imgArray,scale):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    return ver