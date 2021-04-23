import numpy as np
from cv2 import cv2 as cv

def redContours (rect,frame2) :
        lower_red = np.array([0,50,50])
        upper_red = np.array([10,255,255])
        mask2 = cv.inRange(rect, lower_red, upper_red)
        redcnts= cv.findContours(mask2,
                                cv.RETR_TREE,
                                cv.CHAIN_APPROX_SIMPLE)[-2]
        if redcnts :
            red_area = max(redcnts, key=cv.contourArea)
            (x, y, w, h) = cv.boundingRect(red_area)
            imageFrame = cv.rectangle(frame2, (x, y), 
                                    (x + w, y + h), 
                                    (0, 0, 255), 2)
              
            cv.putText(imageFrame, "Red Colour", (x, y),
                        cv.FONT_HERSHEY_DUPLEX, 1.0,
                        (0, 0, 255))

cap = cv.VideoCapture(0)

while 1:
    _, frame = cap.read()
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lower_blue = np.array([100,150,0])
    upper_blue = np.array([140,255,255])
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    bluecnts = cv.findContours(mask,
                              cv.RETR_EXTERNAL,
                              cv.CHAIN_APPROX_SIMPLE)[-2]
    if bluecnts :
        blue_area = max(bluecnts, key=cv.contourArea)
        (x, y, w, h) = cv.boundingRect(blue_area)
        imageFrame = cv.rectangle(frame, (x, y), 
                                (x + w, y + h), 
                                (0, 255, 0), 2)
        cv.putText(imageFrame, "Warna Biru", (x, y),
                        cv.FONT_HERSHEY_DUPLEX, 1.0,
                        (0, 255, 0))
        rect = hsv[y:y+h,x:x+w]
        redContours(rect,imageFrame)

    cv.imshow('frame',frame)
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

cap.release()
cv.destroyAllWindows()
