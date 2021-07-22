import cv2
import numpy as np

img = cv2.imread('dataset/6.jpg')
img = cv2.resize(img,(450,350))
blur = cv2.GaussianBlur(img,(3,3),3)
gray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


canny_edges = cv2.Canny(gray,127,255)
dilate = cv2.dilate(canny_edges,(5,5),5)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
closing = cv2.morphologyEx(canny_edges, cv2.MORPH_CLOSE, kernel)
contours,_ = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    epsilon = 0.05*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    area = cv2.contourArea(cnt)
    
    if len(approx) == 4 and area >120 and area < 124:
        print(area)
        (x,y,w,h) = cv2.boundingRect(cnt)
        cv2.rectangle(img,(x,y),(w,h),(0,255,0),1)

        # print(rect_shape)
cv2.imshow("blur",dilate)
cv2.imshow("Img",img)
cv2.imshow("closing",closing)


cv2.waitKey(0)
cv2.destroyAllWindows()
