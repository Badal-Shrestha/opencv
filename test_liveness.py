import pickle
import cv2
import numpy as np 
from skimage.feature import local_binary_pattern
 
modelFile = "models/dnn/res10_300x300_ssd_iter_140000.caffemodel" 
configFile = "models/dnn/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
classes = ['fake','real'] # Classification Category
img  = cv2.imread("dataset/obama2.jpg") # image to classify

# loading livness model 
with open('models/svm/liveness.pkl','rb') as outfile: 
    liveness_model = pickle.load(outfile) 

"""
    Note: This model is trained on very few dataset so accuracy will be low 
    For better accuracy increase the training dataset 
"""


def detect_face(img, threshold=0.85): # face detection
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


def lbp_histogram(color_image): 
    """
        Return lbp pattern of an  image 
    """
    img =cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY)
    patterns = local_binary_pattern(img, 12, 3) # Neighbors = 12 Radius =3

  
    return patterns


bounding_box = detect_face(img) #detecting face 
for box in bounding_box:
    (x1, y1, x2, y2) = box.astype("int")
    crop_face = img[y1:y2,x1:x2] # Croping face
    crop_face = cv2.resize(crop_face,(160,160))

    lbp_feature = lbp_histogram(crop_face) # calulating image lbp image feature
    w,h,_ = crop_face.shape
    features = lbp_feature.reshape((1,w*h))  # resizing image to match the model  input format
    result = liveness_model.predict_proba(features) # prediction class  reutrn accuracy of class
    print(result)

    index = np.argmax(result[0]) # index of highest accuracy value i.e 1 [[0.40014934 0.59985066]]
    cls = classes[index]  # get Category name using index

    cv2.putText(img,cls,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2) # put text on image

    cv2.imshow("Frame",img)
    cv2.waitKey()
    cv2.destroyAllWindows() 

