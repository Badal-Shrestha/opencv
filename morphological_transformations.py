import cv2

img = cv2.imread("dataset/id.jpg")
blur = cv2.GaussianBlur(img,(3,3),3)
gray = cv2.cvtColor(blur , cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)
adaptive_thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
errode = cv2.erode(thresh,(3,3),3)
cv2.imshow("Adaptive thres",adaptive_thresh)
cv2.imshow("errode",errode)

cv2.waitKey(0)
cv2.destroyAllWindows()