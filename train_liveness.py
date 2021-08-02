import os 
import numpy as np
import cv2
import pickle
from tqdm import tqdm
from sklearn.svm import SVC
from skimage.feature import local_binary_pattern

def lbp_histogram(color_image):
    # color_image = cv2.imread(color_image)
    img =cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY)
    patterns = local_binary_pattern(img, 12, 3)
  
    return patterns


features = [] 
labels = []
image_size = 160 
dataset = "dataset/liveness/" # image path


for root, files, images in os.walk(dataset):
    """
        creating dataset by calulating LBP
        The features are stored in features list
        and labels are stored in labels list

        labels are store in terms  of 0 and 1 
        0 if fake and 1 if real 
    """
    class_name = root.split("/")
    print(class_name,len(class_name))
    if len(class_name) != 3:
        continue

    i =0
    for image in tqdm (images):
        img_path = os.path.join(root,image)
        img = cv2.imread(img_path)
        img = cv2.resize(img,(image_size,image_size))
        lbph_feature = lbp_histogram(img)
        
        features.append(lbph_feature)

        if class_name[-1] =="fake": #set label to 0 if class is fake and 1 if class is real
            labels.append(0)
        else:
            labels.append(1)
     
features = np.array(features)

classifier =SVC(kernel='linear', probability=True, C=10, gamma=0.001,tol=0.5) # Creating model objet for training  

no_sample, w, h = features.shape
print(no_sample, w, h )

# Reshaping the features to 1D because the SVM model take 1D formate
features = features.reshape((no_sample,w*h)) 

labels = np.array(labels)

print ("Training features: {}".format(features[0].shape), labels.shape)

classifier.fit(features, labels) # training model

with open("models/svm/liveness.pkl", 'wb') as outfile: #saving model
    pickle.dump(classifier, outfile)