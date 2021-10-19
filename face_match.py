import cv2
import time
import face_models
import numpy as np 
import facenet
from face_models import detect_face 
from scipy import misc
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances



input_image_size = 160

model = face_models.LoadRecogModel()
embedding_size = model.embedding_tensor()
facemodel = face_models.LoadFaceModel()

pnet, rnet, onet = facemodel.nets()
minsize = 75  # minimum size of face
threshold = [0.5, 0.6, 0.7]  # three steps's threshold
factor = 0.709  # scale factor
image_size = 182
emb_array = np.zeros((1, embedding_size))

def get_embeding(img):
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    det = bounding_boxes[:, 0:4]

    cropped = []
    scaled = []
    scaled_reshape = []
    nrof_faces = len(bounding_boxes)
    bb = np.zeros((nrof_faces, 4), dtype=np.int32)
    if nrof_faces > 0:
        for i in range(nrof_faces):
            emb_array = np.zeros((1, embedding_size))

            bb[i][0] = det[i][0]
            bb[i][1] = det[i][1]
            bb[i][2] = det[i][2]
            bb[i][3] = det[i][3]

            # inner exception
            if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(img[0]) or bb[i][3] >= len(img):
                continue

            cropped.append(img[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
            cropped[i] = facenet.flip(cropped[i], False)
            # print(cropped)
            scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
            scaled[i] = cv2.resize(scaled[i], (input_image_size, input_image_size),
                                    interpolation=cv2.INTER_CUBIC)
            scaled[i] = facenet.prewhiten(scaled[i])
            scaled_reshape.append(scaled[i].reshape(-1, input_image_size, input_image_size, 3))

            

            #Call function inside the loaded model
            predictions = model.predict(scaled_reshape[i], emb_array)
            # predictions = model.predict(scaled_reshape, emb_array)
        return predictions
    else:
        return []


def get_matching(img1,img2):
    emb1 = get_embeding(img1)
    emb2 = get_embeding(img2)
    print(emb1.shape,emb2.shape)
    similarity_score = cosine_similarity(emb1,emb2)
    print(similarity_score,time.time()-start_time)

    # score = float('{:.2f}'.format(similarity_score[0][0])) 
    # if similarity_score >= 0.6:
    #     title = f"Matched :{score*100}%"
    # else:
    #     title = f"Matched :{score*100}%"



if __name__ == "__main__":
    start_time = time.time()
    img1 = cv2.imread("dataset/citizen.jpg")
    img2 = cv2.imread("dataset/test.jpg")

    get_matching(img1,img2)
    print(time.time()-start_time)
