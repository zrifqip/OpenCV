import numpy as np
from cv2 import cv2 as cv

def redContours (rect) :
        lower_red = np.array([0,50,50])
        upper_red = np.array([10,255,255])
        mask2 = cv.inRange(rect, lower_red, upper_red)
        contours= cv.findContours(mask2,
                                           cv.RETR_TREE,
                                           cv.CHAIN_APPROX_SIMPLE)
      
        for contour in enumerate(contours):
            area = cv.contourArea(contour)
            if(area > 300):
                x, y, w, h = cv.boundingRect(contour)
                imageFrame = cv.rectangle(imageFrame, (x, y), 
                                        (x + w, y + h), 
                                        (0, 0, 255), 2)
              
                cv.putText(imageFrame, "Red Colour", (x, y),
                            cv.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 0, 255))

cap = cv.VideoCapture(0)

while 1:
    _, frame = cap.read()
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([100,150,0])
    upper_blue = np.array([140,255,255])
    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    bluecnts = cv.findContours(mask,
                              cv.RETR_EXTERNAL,
                              cv.CHAIN_APPROX_SIMPLE)[-2]
    for pic, contour in enumerate(bluecnts):
        area = cv.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv.boundingRect(contour)
            imageFrame = cv.rectangle(frame, (x, y), 
                                       (x + w, y + h), 
                                       (0, 255, 0), 2)
            cv.putText(imageFrame, "Warma Biru", (x, y),
                        cv.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 0))
            redContours(contour)
        

    cv.imshow('frame',frame)
    cv.imshow('mask',mask)

     
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

cap.release()
cv.destroyAllWindows()