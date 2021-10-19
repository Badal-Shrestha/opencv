import cv2
import base64


img = cv2.imread("static/photo.jpg")
cv2.imshow("img",img)
cv2.waitKey(0)

# cv_string = base64.encodebytes(img)

retval, buffer_img= cv2.imencode('.jpg', img)
data = base64.b64encode(buffer_img)



print(data)

with open("base.txt",'wb')as outfile:
    outfile.write(data)