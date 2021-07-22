import cv2

face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

img= cv2.imread("dataset/16.jpg")
img = cv2.resize(img,(450,350))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray,1.3,5)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    cv2.imshow("Crop",roi_color)


cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


""" 
******************************* For video **********************

cap = cv2.VideoCapture("dataset/189.mp4")
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame,(350,450))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
    

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

"""
