import cv2
from mtcnn import MTCNN

img= cv2.imread("dataset/3.jpg")
img = cv2.resize(img,(450,350))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_detector = MTCNN()

face_data = face_detector.detect_faces(img)
print(face_data)

for faces in face_data:
    bounding_box = faces.get("box")
    x = bounding_box [0]
    y = bounding_box [1]
    w = bounding_box [2] + x
    h = bounding_box [3] + y
    # crop = img[y:h,x:w]
    # cv2.imshow("Crop",crop)
    # cv2.waitKey(0)

    cv2.rectangle(img,(x,y),(w,h),(0,255,0),1)
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

""" 
******************************* For video **********************

cap = cv2.VideoCapture("dataset/189.mp4")
face_detector = MTCNN()
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame,(350,450))
    face_data = face_detector.detect_faces(frame)
    # print(face_data)

    for faces in face_data:
        bounding_box = faces.get("box")
        x = bounding_box [0]
        y = bounding_box [1]
        w = bounding_box [2] + x
        h = bounding_box [3] + y
        # crop = frame[y:h,x:w]
        # cv2.imshow("Crop",crop)
        # cv2.waitKey(0)

        cv2.rectangle(frame,(x,y),(w,h),(0,255,0),1)
  
    

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

"""
