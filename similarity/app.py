from flask import Flask, request, render_template, send_from_directory, jsonify
from keras_cosine_similarity import check_similarity
import numpy as np
import cv2 
import base64
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)
# ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
@app.route("/", methods=["POST",'GET'])
@cross_origin()
def home():
    return render_template('home.html')


def data_uri_to_cv2_img(uri):
    """
        Convert Base64 encoded to numpy opencv readable
    """
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

@app.route("/similarity", methods=["POST",'GET'])
@cross_origin()
def similarity():
    """
        Calculate Image similarity
        return image similarity score and Matched, Not Matched label depending in threshold

        return sample {
                score:["0.6","Matched"]
            }
    """
    if request.method == 'POST':
        
        for upload in request.files.getlist("file"):
            print("{} is the file name".format(upload.filename))
            img_array = np.array(bytearray(upload.read()), dtype=np.uint8)

            img1 = cv2.imdecode(img_array, -1)

        img2 = request.form['img2'] # reading img 2

        img2 = data_uri_to_cv2_img(img2) 
        
        score = check_similarity(img1,img2) # Return similarity score
        print(score)
        output = "Matched"  if score >0.6 else "Not Matched" # Marched if similarity socre is greater than 0.6 else Not matched
       

       
    data ={'score': [str(score),output]}
    print(data)
    return jsonify(data)

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True, use_reloader=False, port=5000,ssl_context='adhoc')