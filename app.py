# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 07:27:44 2020

@author: Jagadheeswar Reddy
"""
# coding=utf-8
import os,shutil

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


from NASNetLarge_Test import NASNetLarge_Test
from ResNet50_Test import ResNet50_Test 
from Resnet101V2_Test import Resnet101V2_Test 
from VGG19_Test import VGG19_Test 
from collections import OrderedDict
import json

app = Flask(__name__)

def predict(file_path):
    nasNet_name,nasNet_accurary = NASNetLarge_Test.model_nasNetLarge_predict(file_path)  
    resNet50_name,resNet50_accurary = ResNet50_Test.model_resnet50_predict(file_path)  
    resNet101_name,resNet101_accurary = Resnet101V2_Test.model_resnet101V2_predict(file_path)  
    vgg19_name,vgg19_accurary = VGG19_Test.model_vgg19_predict(file_path)
    
    keys=[nasNet_accurary,resNet50_accurary,resNet101_accurary,vgg19_accurary]
    values=[nasNet_name,resNet50_name,resNet101_name,vgg19_name]
    classificationResult = dict(zip(keys, values))
    sortedReult = OrderedDict(sorted(classificationResult.items(), key=lambda t: t[0],reverse=True))
    
    best_accuracy=list(sortedReult.keys())[0]
    best_name=list(sortedReult.values())[0]
    
    result=dict()
    result["NASNetLarge"]="{0},{1}".format(nasNet_name,nasNet_accurary)
    result["ResNet50"]="{0},{1}".format(resNet50_name,resNet50_accurary)
    result["Resnet101V2"]="{0},{1}".format(resNet101_name,resNet101_accurary)
    result["VGG19"]="{0},{1}".format(vgg19_name,vgg19_accurary)
    result["Finalized"]="{0},{1}".format(best_name,best_accuracy)
    return result


@app.route('/')
def home():
    return render_template('index.html',title="ImageClassification")
@app.route('/index.html')
def index():
    return render_template('index.html',title="ImageClassification")
@app.route('/about.html')
def about():
    return render_template('about.html',title="About")
@app.route('/face_recognition.html')
def review():
    return render_template('face_recognition.html',title="FaceRecognition")
@app.route('/contact.html')
def contact():
    return render_template('contact.html',title="Contact")


@app.route('/predict', methods=['POST'])
def upload():
        basepath = os.path.dirname(__file__)
        folder_path=os.path.join(basepath, 'uploads')
        # remove all files under uploads
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
                
        # Get the file from post request
        f = request.files['file']
        # Save the file to ./uploads
        path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(path)
        result_json= json.loads(json.dumps(predict(path)))
        return result_json

if __name__ == '__main__':
   app.run(port=5003)