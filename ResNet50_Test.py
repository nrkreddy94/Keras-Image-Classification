# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 07:27:44 2020

@author: Jagadheeswar Reddy
"""

# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
#from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50

class ResNet50_Test:
    
    #MODEL_PATH = 'model_resnet.h5'
    # Load your trained model
    #model_resnet50 = load_model(MODEL_PATH)
    model_resnet50=  ResNet50(weights='imagenet')
   

    @staticmethod
    def model_resnet50_predict(img_path, model=model_resnet50):
        img = image.load_img(img_path, target_size=(224, 224))
    
        # Preprocessing the image
        x = image.img_to_array(img)
        # x = np.true_divide(x, 255)
        x = np.expand_dims(x, axis=0)
    
        # Be careful how your trained model deals with the input
        # otherwise, it won't make correct prediction!
        x = preprocess_input(x)
    
        preds = model.predict(x)
        pred_class = decode_predictions(preds, top=1)
        name=pred_class[0][0][1]
        accuracy=pred_class[0][0][2]
        return name,accuracy






