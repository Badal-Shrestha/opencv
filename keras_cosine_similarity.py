import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances



img1 = cv2.imread("dataset/obama1.jpg") # read image
img2 = cv2.imread("dataset/obama2.jpg")


# Model for face detection
modelFile = "models/dnn/res10_300x300_ssd_iter_140000.caffemodel" 
configFile = "models/dnn/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Facenet model to extract the facial feature from the image 
model = load_model('models/facenet_keras.h5') # loading model
# model.summary() # Display Architecture of the model

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

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  


def get_img_feature(img):
    """
        Extract face feature from the image using facenet model 
        model input shape: 160x160
    """
    bounding_box = detect_face(img)
    for box in bounding_box:
        (x1, y1, x2, y2) = box.astype("int")
        crop_face = img[y1:y2,x1:x2]
        crop_face = cv2.resize(crop_face,(160,160))
        crop_face = prewhiten(crop_face)
        embeding = model.predict(np.expand_dims(crop_face,axis=0))
    return embeding




emb1 = get_img_feature(img1)
emb2 = get_img_feature(img2)

similarity_score = cosine_similarity(emb1,emb2) # Comapring image using Consine Similarity algorithm

if similarity_score >= 0.6:
    print("*********** Matched :",similarity_score, " ********************")
else:
    print("**************** Not matched **************")
