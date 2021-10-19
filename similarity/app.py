from flask import Flask, request, render_template, send_from_directory, jsonify
from flask.helpers import url_for
from keras_cosine_similarity import check_similarity
from werkzeug.utils import redirect, secure_filename
import numpy as np
import cv2 
import base64
from db import db_init, db
from models import Img

from flask_cors import CORS, cross_origin

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///img.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db_init(app)

CORS(app)
# ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
@app.route("/", methods=["POST",'GET'])
@cross_origin()
def home():
    return render_template('home.html')


@app.route("/redirect",methods=["POST",'GET'])
def redirect_page():
    # return redirect(url_for(".home"))
    data = {"sucess": True, "cid":10000}
    return jsonify(data) 


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
        print("Insied-----------")
        pic = request.files['file']
        for upload in request.files.getlist("file"):
            print("{} is the file name".format(upload.filename))
            img_array = np.array(bytearray(upload.read()), dtype=np.uint8)

            img1 = cv2.imdecode(img_array, -1)

        img2 = request.form['img2'] # reading img 2

        img2 = data_uri_to_cv2_img(img2) 
        cv2.imshow("Selfi",img2)
        cv2.waitKey(0)
        
        score = check_similarity(img1,img2) # Return similarity score

        filename = secure_filename(pic.filename)

        mimetype = pic.mimetype
        if not filename or not mimetype:
            return 'Bad upload!', 400

        img = Img(img=pic.read(), name=filename, mimetype=mimetype)
        db.session.add(img)
        db.session.commit()

        print(score)
        output = "Matched"  if score >0.6 else "Not Matched" # Marched if similarity socre is greater than 0.6 else Not matched
       

       
    data ={'score': [str(score),output]}
    print(data)
    return jsonify(data)

@app.route('/display/<int:id>')
def display(id):
    img = Img.query.filter_by(id=id).first()
    if not img:
        return 'Img Not Found!', 404
    print(img.id,img.img)
    nparr = np.fromstring(img.img, np.uint8)
    img_np = cv2.imdecode(nparr,flags=1) 
   
    retval, buffer = cv2.imencode('.jpg', img_np)
    jpg_as_text = base64.b64encode(buffer)
    jpg_as_text = jpg_as_text.decode("utf-8") 

    return render_template("display.html",image=jpg_as_text)

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True, use_reloader=False, port=5000,ssl_context='adhoc')