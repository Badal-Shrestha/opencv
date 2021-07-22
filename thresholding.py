import cv2

img = cv2.imread("dataset/2.png")
blur = cv2.blur(img,(3,3))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(img,220,255)

ret,thresh = cv2.threshold(gray,220,255,cv2.THRESH_BINARY)

# Adaptive threshold
adaptive_thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2)
# adaptive_thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

""" Otsu Thresholding """

"""Finding contours """
contours,_ = cv2.findContours(adaptive_thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img,contours,-1,(255,0,0),2) # Drawing countours in image

cv2.imshow("thresh",thresh)
cv2.imshow("img",img)
cv2.imshow("canny",adaptive_thresh)


cv2.waitKey(0)
cv2.destroyAllWindows()