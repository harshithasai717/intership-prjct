import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template,redirect,flash,url_for
#from werkzeug import secure_filename

UPLOAD_FOLDER = "static/uploads"
#ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['IMAGE_UPLOADS'] = UPLOAD_FOLDER
#app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

rcnn_model = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'faster_rcnn_inception_v2_coco_2018_01_28.pbtxt')
type_model = load_model("appy.hdf5")
classes = ["Green_Apple","Red_Apple","Rotten_Apple"]

def pre(img_path):
    # Input image
    img = cv2.imread(img_path)
    rows, cols, channels = img.shape

    # Use the given image as input, which needs to be blob(s).
    blob_img = cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False)
    rcnn_model.setInput(blob_img)

    # Runs a forward pass to compute the net output
    networkOutput = rcnn_model.forward()

    # Loop on the output `s5|t40=
    for detection in networkOutput[0,0]:
        score = float(detection[2])
        if score > 0.2001:
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            rcnn_pred_img = img[int(top):int(bottom),int(left):int(right)]

    # adjusting the predicted apple image form faster rcnn inception v2
    rcnn_pred_img = cv2.cvtColor(rcnn_pred_img,cv2.COLOR_BGR2RGB)
    rcnn_pred_img = cv2.resize(rcnn_pred_img, (100,100))

    # pre_processing the apple image to predict the type
    rcnn_pred_img = np.array(rcnn_pred_img)
    rcnn_pred_img = rcnn_pred_img.reshape(1,100,100,3)
    type_pred = type_model.predict(rcnn_pred_img)

    # returning the final ans in terms of classes string
    return classes[type_pred.argmax()]

app=Flask(__name__)
app.secret_key="secure"
app.config['UPLOAD_FOLDER'] = str(os.getcwd())+'/static/uploads'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/',methods=["post","get"])
def first_page():
    if request.method=="POST":
        global image_name,image_data

        file = request.files['file']
        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            op = pre('static/uploads/'+filename)
            # solution = SOLUTIONS[op]
            return render_template("data_page.html",filename=filename, result = op)
        else:
            flash('Allowed image types are -> png, jpg, jpeg, gif')
            return redirect(request.url)

    else:
        return render_template("form_page.html")


@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == "__main__":
    
    app.run(host="0.0.0.0",port=5000, debug=True) 
