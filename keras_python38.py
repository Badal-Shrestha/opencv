import cv2
import numpy as np
import time
from numpy.lib.index_tricks import MGridClass
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from keras_facenet import FaceNet
import matplotlib.pyplot as plt

start_time = time.time()

img1 = cv2.imread("dataset/citizen.jpg") # read image
img2 = cv2.imread("dataset/obama1.jpg")

matched_data = []

# Model for face detection
modelFile = "models/dnn/res10_300x300_ssd_iter_140000.caffemodel" 
configFile = "models/dnn/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
# Facenet model to extract the facial feature from the image 
model = FaceNet()
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
        matched_data.append(cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),1))
        # crop_face = prewhiten(crop_face)

        embeding = model.embeddings(np.expand_dims(crop_face,axis=0))
    return embeding




emb1 = get_img_feature(img1)
emb2 = get_img_feature(img2)

similarity_score = cosine_similarity(emb1,emb2) # Comapring image using Consine Similarity algorithm
print("Endtime :: " , time.time() - start_time)
print("simgg",similarity_score[0][0])
score = float('{:.2f}'.format(similarity_score[0][0])) 
print( type(score),score *100)
if similarity_score >= 0.6:
    title = f"Matched :{score*100}%"
else:
    title = f" Not Matched :{score*100}%"

fig = plt.figure(figsize=(150,150))
fig.suptitle(title, fontsize=20)
for i in range(0,len(matched_data)):
    # fig.add_subplot()
    fig.add_subplot(1, 2, i+1)
    plt.imshow(matched_data[i])
plt.show()
# for i , img in enumerate (matched_data):
#     cv2.imshow(f"img{i}",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

