import cv2

img = cv2.imread("dataset/2.png")
# blur = cv2.blur(img,(3,3))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #0-255

ret,thresh = cv2.threshold(gray,220,255,cv2.THRESH_BINARY) # p> 220 ==255 nad if p<220 it 0
cv2.imshow("thresh",thresh)
canny = cv2.Canny(img,220,255)
# Adaptive threshold
adaptive_thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2)
# adaptive_thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
cv2.imshow("adaptive_thresh",adaptive_thresh)

""" Otsu Thresholding """

"""Finding contours """
contours,_ = cv2.findContours(adaptive_thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img,contours,-1,(255,255,0),3) # Drawing countours in image


cv2.imshow("img",img)
# cv2.imshow("canny",canny)   


cv2.waitKey(0)
cv2.destroyAllWindows()


