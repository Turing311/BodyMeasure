import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from CenterHMR.core.test import Demo
# Some utilites
import numpy as np
from util import base64_to_pil

from CenterHMR.core.test import main

# Declare a flask app
app = Flask(__name__)
demo = Demo()
# result = demo.run('../_tmp/in', 150)
# print(result)

print('Model loaded. Check http://127.0.0.1:5000/')


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        image = base64_to_pil(request.json['image'])
        height = int(request.json['height'])

        if not os.path.exists('../_tmp/in'):
            tf.gfile.MakeDirs('../_tmp/in')

        image.save('../_tmp/in/work.png')
	
        # res_im,seg=MODEL.run(image)

        # seg=cv2.resize(seg.astype(np.uint8),image.size)
        # mask_sel=(seg==15).astype(np.float32)
        # mask = 255*mask_sel.astype(np.uint8)

        # img = 	np.array(image)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)   

        # res = cv2.bitwise_and(img,img,mask = mask)
        # bg_removed = res + (255 - cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)) 
        # result = main(bg_removed,height,None)
        
        result = demo.run('../_tmp/in', height)
        obj = ''
        with open('test.obj', 'r') as file:
            obj = file.read()
        return jsonify(result=result, obj=obj)
		

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)
    print("main start")
    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    print("main end")
    http_server.serve_forever()
    print("main end1")


