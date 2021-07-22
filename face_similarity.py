import cv2
import numpy as np
from deepface import DeepFace


result = DeepFace.verify("dataset/obama1.jpg","dataset/obama2.jpg",model_name='VGG-Face')
print(result)