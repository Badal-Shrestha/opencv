import cv2
import numpy as np
import video_active_approach
from collections import Counter
# face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
# ds_factor=0.6
modelFile = "../models/dnn/res10_300x300_ssd_iter_140000.caffemodel" # Model path
configFile = "../models/dnn/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

def detect_face(img, threshold=0.85):
    """
        Detect Face 
    """
    height, width = img.shape[:2]
        
    blob = cv2.dnn.blobFromImage(img,1.0, (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces3 = net.forward()

    boxes = []
    for i in range(faces3.shape[2]):
        confidence = faces3[0, 0, i, 2]
        if confidence > threshold:
            box = faces3[0, 0, i, 3:7]* np.array([width, height, width, height])
            boxes.append(box)
    return np.array(boxes)

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.width = self.video.get(3)
        self.height = self.video.get(4)
        self.fx, self.fy , self.fx1, self.fy1 = np.array((self.width*0.3, self.height*0.2, self.width*0.7, self.height*0.8)).astype("int")
        self.face_frame_size = (self.fx1-self.fx)* (self.fy1-self.fy)

        self.processing_frame = {'front': [], 'right': [], 'left': []}
        self.frame_counter = 2 # number frame to consider for each front, left , right profile prediction
        self.frame_to_save = []
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, frame = self.video.read()
        guide_txt = "Position Your Face Inside Frame"
        box_color = (0,0,255)
            
        front = self.processing_frame.get("front")
        left = self.processing_frame.get("left")
        right = self.processing_frame.get("right")
        frame = cv2.flip(frame,3)

        frame1 = frame.copy()
        # image = cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE)
        # bounding_box = detect_face(frame)
        faces, keypoints = video_active_approach.frame_grabber_keypoints(frame)
        if len(faces) == 0:
            self.processing_frame = {'front': [], 'right': [], 'left': []}
            self.frame_to_save.clear()
            print("NO Face found!!!!")

        elif len(faces) > 1:
            self.processing_frame = {'front': [], 'right': [], 'left': []}
            self.frame_to_save.clear()
            cv2.putText(frame,"Multiple Faces Not Acceptable",(50,50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,255),2)
        
        else:
            face_box = faces[0].astype("int").tolist()
            (x,y,x1,y1) = face_box
            frame = cv2.rectangle(frame, (x,y), (x1,y1), (255,0,0), 1)
            face = frame1[y:y1, x:x1]

            if self.fx < x and self.fy < y and x1 < self.fx1 and y1 < self.fy1 and (face.size / self.face_frame_size) > 0.5: 

                if len(self.frame_to_save) == 0 and len(front) == self.frame_counter -1:
                    self.frame_to_save.append(frame1)
                    print("frame to save::",len(self.frame_to_save))
                
                box_color = (0,255,0)
                guide_txt = "Slightly Roate your head"

                prediction = video_active_approach.passive_liveness(frame1, face_box)
                # draw result of prediction
                label = np.argmax(prediction)
                value = prediction[0][label]/2
                
                if value >= 0.5:
                    if len(front) < self.frame_counter:
                        front.append(label)
                    
                    hpose = video_active_approach.detect_pose(keypoints)
                
                    if hpose == "left" and len(left) < self.frame_counter:
                        left.append(label)
                    
                    if hpose == "right" and len(right) < self.frame_counter:
                        right.append(label)
                
                    frame = cv2.putText(frame, hpose, (self.fx,self.fy-5), 1,2,(0,255,255),2)
            else:
                self.processing_frame = {'front': [], 'right': [], 'left': []}
                self.frame_to_save.clear()

                
        if len(front) == self.frame_counter and len(left) == self.frame_counter and len(right) == self.frame_counter and len(self.frame_to_save)==1:
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
            print("frame to save::",len(self.frame_to_save))
            cv2.imshow("Frma",self.frame_to_save[0])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            self.frame_to_save = cv2.putText(self.frame_to_save[0],predict_txt , (50,50), 1,2,txt_color,2)
            

        frame = cv2.rectangle(frame, (self.fx,self.fy), (self.fx1,self.fy1),box_color,3)
        frame = cv2.putText(frame, guide_txt, (self.fx - 30,self.fy1 + 20), 1,1,box_color,1)

        # for box in bounding_box:
        #     (x1, y1, x2, y2) = box.astype("int")       
        #     cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
