import cv2
import numpy as np

img = cv2.imread("/media/mrcloud/New Volume/Projects/ocr/images/download.jpeg")
img = cv2.resize(img,(450,350))
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

filter_kernel_horizontal = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

filter_kernel_vertical = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

filtered_img_horizontal = cv2.filter2D(gray,-1,filter_kernel_horizontal)
filtered_img_vertical = cv2.filter2D(gray,-1,filter_kernel_vertical)

rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
merge = img_vh = cv2.addWeighted(filtered_img_vertical,0.5,filtered_img_horizontal,0.5,0.0)
ret = cv2.filter2D(merge,-1,rect_kernel)
cv2.imshow("img", img)

cv2.imshow("Filtered", filtered_img_horizontal)
cv2.imshow("Filtered vertical", filtered_img_vertical)

cv2.imshow("Merged",merge)
cv2.imshow("rect", ret)
cv2.waitKey(0)
cv2.destroyAllWindows()
