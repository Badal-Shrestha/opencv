import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern
      
"""

    email : badal.shrestha@infodevelopers.com.np

"""

def lbp_histogram(color_image):
    # color_image = cv2.imread(color_image)
    img =cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY)
    patterns = local_binary_pattern(img, 12, 3)
  
    return patterns


if __name__ == "__main__":
    real_path = cv2.imread("dataset/real1.jpg")
    real_path = cv2.resize(real_path,(160,160))
    fake_path = cv2.imread('dataset/fake1.jpg')
    fake_path = cv2.resize(fake_path,(160,160))

    # cv2.imshow("real",lbp_histogram(real_path))
    cv2.imshow("real",lbp_histogram(real_path))
    cv2.imshow("fake--",lbp_histogram(fake_path))

    
    cv2.waitKey(0)
