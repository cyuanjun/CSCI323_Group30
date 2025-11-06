import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from proj_utils import *
#import proj_sudokuSolver

#####################################
pathImage = "Resources/7.png"
heightImg = 450
widthImg = 450
#model = initlizePredictionModel()
#####################################

#### 1. Prepare the image
img = cv2.imread(pathImage)
img = cv2.resize(img, (widthImg, heightImg))
imgBlank = np.zeros((heightImg, widthImg), np.uint8)
imgThreshold = preProcess(img)

#### 2. Find all contours
imgContours = img.copy()  # Copy image for display purposes
imgBigContour = img.copy()  # Copy image for display purposes
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgContours, contours, -1,  (0, 255, 0), 3)  # Draw all detected contours


#### 3. Find the biggest contour
biggest, maxArea = biggestContour(contours)  # Find the biggest contour
print(biggest)
if biggest.size != 0:
    biggest = reorder(biggest)
    print(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0,0,255), 25)  # Draw with the biggest contours
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgDetectedDigits = imgBlank.copy()
    imgWarpColored = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)

    #### 4. Split the image
    imgSolvedDigits = imgBlank.copy()
    boxes = splitBoxes(imgWarpColored)
    print(len(boxes))








imageArray = ([img,imgThreshold,imgContours, imgBigContour],
                  [imgDetectedDigits, imgBlank, imgBlank, imgBlank])
stackedImage = stackImages(imageArray, 1)


cv2.imshow('Stacked Images', stackedImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
