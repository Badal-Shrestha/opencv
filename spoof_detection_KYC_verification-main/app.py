from flask import Flask, request, render_template, send_from_directory, jsonify, Response
from werkzeug.utils import secure_filename
import numpy as np
import cv2 
import base64
from camera import VideoCamera
import video_active_approach
import math
from flask_cors import CORS, cross_origin

app = Flask(__name__)

CORS(app)




def data_uri_to_cv2_img(uri):
    """
        Convert Base64 encoded to numpy opencv readable
    """
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

@app.route("/", methods=["POST",'GET'])
@cross_origin()
def liveVideo():
    return render_template('video.html')


@app.route("/videoApproach", methods=["POST",'GET'])
@cross_origin()
def videoApproach():
    """
        Calculate Image similarity
        return image similarity score and Matched, Not Matched label depending in threshold

        return sample {
                score:["0.6","Matched"]
            }
    """
    if request.method == 'POST':
        

        frame = request.form['video'] # reading img 2
        frame = data_uri_to_cv2_img(frame)
        bounding_box = request.form['bounding_box']
        bb_box = bounding_box.split(",")
        bb_box = list(map(int, bb_box))
        prediction = video_active_approach.passive_liveness(frame,bb_box)
        label = np.argmax(prediction)
        value = prediction[0][label]/2
       
        print(prediction, int( label), value)
        
    
    return jsonify([int(label),value])


if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True, use_reloader=False, port=5000,ssl_context='adhoc')