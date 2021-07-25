from keras_cosine_similarity import calulate_smilarity



@route("/similarity")
def similariy():
    img1 = get_js()
    img2 = get_js()
    similariyty =calulate_smilarity(img1,img2)
    return similariyty

