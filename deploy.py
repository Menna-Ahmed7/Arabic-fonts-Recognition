from flask import Flask, request,jsonify
import pandas as pd
from flask_cors import CORS
from os.path import join, dirname, realpath
import json
import glob
import math
import os
import datetime
from flask import request
from PIL import Image
from lpq import lpq
import pickle
import cv2
import numpy as np
from preprocessingunit import process_image,preprocess
from PIL import Image

app = Flask(__name__)
CORS(app)


# model=pickle.load("Logistic Regression.pkl")
with open('Logistic Regression.pkl', 'rb') as file:
    # Deserialize and load the object from the file
    model = pickle.load(file)

# app.config['PHOTO_PATH'] = join(dirname(realpath(__file__)), 'Photos')

@app.route("/image_3", methods=["POST"])
def imagePrediction():
    file=request.files['image']
    img=cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    # img=Image.open(file.stream)
    # img=Image.open(file.stream)
    # img_path = join(app.config['PHOTO_PATH'], 'uploaded_image.jpeg')
    # print(img) 
    # # Process the image
    proceseed_img=preprocess(img, True)
    proceseed_img = Image.fromarray(proceseed_img)  # Convert to PIL Image object
    proceseed_img.save('uploaded_image.jpeg')

    # # # # Load the preprocessed image (assuming it's saved)
    # path="Processed-test/test/"+img
    # image = cv2.imread(path)

    # # # # Extract LPQ features or other processing if needed
    
    imageLPQ = lpq('uploaded_image.jpeg')  # Replace with your specific feature extraction logic
    # # # # Predict the label using your model
    label = model.predict(np.array([imageLPQ]))[0]
    # # return 
    return jsonify({'label': str(label)})

# check if the post request has the file part
    # if 'file' not in request.files:
    #         return "No file found"
    # user_file = request.files['file']
    # if user_file.filename == "":
    #     return "file name not found"
    # else:
    #     path=os.path.join(os.getcwd()+'\\modules\\static\\'+user_file.filename)
    #     user_file.save(path)

# app.run(debug=True)
app.run(host='0.0.0.0', port=5000)
# app.run(host='localhost', port=5000)




