from flask import Flask, request, Response, render_template
from werkzeug.utils import secure_filename

from db import db_init, db
from models import Img
import base64
from PIL import Image
import io
import cv2
import numpy as np
import base64
app = Flask(__name__)
# SQLAlchemy config. Read more: https://flask-sqlalchemy.palletsprojects.com/en/2.x/
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///img.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db_init(app)


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    pic = request.files['pic']
    if not pic:
        return 'No pic uploaded!', 400

    filename = secure_filename(pic.filename)
    mimetype = pic.mimetype
    if not filename or not mimetype:
        return 'Bad upload!', 400
    img = cv2.imread("static/test.jpg")
    retval, buffer_img= cv2.imencode('.jpg', img)
    data = base64.b64encode(buffer_img)
    # cv2.imshow("img",img)
    # cv2.waitKey(0)
    img = Img(img=data, name="filename", mimetype=mimetype)
    db.session.add(img)
    db.session.commit()

    return 'Img Uploaded!', 200


@app.route('/display/<int:id>')
def display(id):
    img = Img.query.filter_by(id=id).first()
    if not img:
        return 'Img Not Found!', 404
    # base64_data = base64.b64decode(img.img)
    # nparr = np.fromstring(base64_data, np.uint8)
    # print(nparr,base64_data,img.img)
    # img_np = cv2.imdecode(nparr,flags=1) 
    # retval, buffer = cv2.imencode('.jpg', img_np)
    # jpg_as_text = base64.b64encode(buffer)
    jpg_as_text= img.img
    jpg_as_text = jpg_as_text.decode("utf-8") 
    return render_template("display.html",image=jpg_as_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5050)