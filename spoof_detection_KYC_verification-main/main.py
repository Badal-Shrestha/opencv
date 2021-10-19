import numpy as np
import cv2
from collections import Counter

import video_active_approach

cap = cv2.VideoCapture(0)
# cap.set(3, 720)
# cap.set(4, 1080)
width = cap.get(3)
height = cap.get(4)
fx, fy , fx1, fy1 = np.array((width*0.3, height*0.2, width*0.7, height*0.8)).astype("int")
face_frame_size = (fx1-fx)* (fy1-fy)

processing_frame = {'front': [], 'right': [], 'left': []}
frame_counter = 2 # number frame to consider for each front, left , right profile prediction
frame_to_save = []

head_direction = "left"
while True:
    guide_txt = "Position Your Face Inside Frame"
    box_color = (0,0,255)
          
    front = processing_frame.get("front")
    left = processing_frame.get("left")
    right = processing_frame.get("right")



    ret, frame = cap.read()
    frame = cv2.flip(frame,3)

    frame1 = frame.copy()

    faces, keypoints = video_active_approach.frame_grabber_keypoints(frame)
    if len(faces) == 0:
        processing_frame = {'front': [], 'right': [], 'left': []}
        frame_to_save.clear()
        print("NO Face found!!!!")

    elif len(faces) > 1:
        processing_frame = {'front': [], 'right': [], 'left': []}
        frame_to_save.clear()
        cv2.putText(frame,"Multiple Faces Not Acceptable",(50,50),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,255),2)
    
    else:
        face_box = faces[0].astype("int").tolist()
        (x,y,x1,y1) = face_box
        frame = cv2.rectangle(frame, (x,y), (x1,y1), (255,0,0), 1)
        face = frame1[y:y1, x:x1]

        if fx < x and fy < y and x1 < fx1 and y1 < fy1 and (face.size / face_frame_size) > 0.5: 

            if len(frame_to_save) == 0 and len(front) == frame_counter -1:
                frame_to_save.append(frame1)
            
            box_color = (0,255,0)
            guide_txt = "Slightly Roate your head"

            print("Face box ::::: ",face_box)
            prediction = video_active_approach.passive_liveness(frame1, face_box)
            # draw result of prediction
            label = np.argmax(prediction)
            value = prediction[0][label]/2
            
            if value >= 0.5:
                if len(front) < frame_counter:
                    front.append(label)
                cv2.putText(frame,"Turn "+head_direction,(50,50),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,255),2)
                hpose = video_active_approach.detect_pose(keypoints)
                if head_direction == 'left' and hpose == "left" and len(left) < frame_counter:
                    left.append(label)
                
                
                if head_direction == 'right' and hpose == "right" and len(right) < frame_counter:
                    right.append(label)

                if len(left) ==  frame_counter:
                    head_direction = "right"
                elif len(right) == frame_counter:
                    head_direction ="Complete"
            
                frame = cv2.putText(frame, hpose, (fx,fy-5), 1,2,(0,255,255),2)
        else:
            processing_frame = {'front': [], 'right': [], 'left': []}
            frame_to_save.clear()

            
    if len(front) == frame_counter and len(left) == frame_counter and len(right) == frame_counter:
        total_label = front + left + right
        
        counter = Counter(total_label)
        max_key = max(counter, key=counter.get)
        print("Counter",counter,max_key,total_label)
        if max_key == 1:
            predict_txt = "Face is Real"
            txt_color = (0,255,0)
        else:
            predict_txt = "Unable to define your Liveness"
            txt_color = (0,0,255)

        cv2.destroyWindow("frame")
        print("Frame t0 save", len(frame_to_save))
        frame_to_save = cv2.putText(frame_to_save[0],predict_txt , (50,50), 1,2,txt_color,2)
        
        cv2.imshow("Real", frame_to_save)
        if cv2.waitKey():
            break
        cv2.destroyWindow("Real")

    frame = cv2.rectangle(frame, (fx,fy), (fx1,fy1),box_color,3)
    frame = cv2.putText(frame, guide_txt, (fx - 30,fy1 + 20), 1,1,box_color,1)

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
