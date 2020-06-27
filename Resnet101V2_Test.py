# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 07:27:44 2020

@author: Jagadheeswar Reddy
"""


from tensorflow.keras.applications.resnet_v2 import preprocess_input, decode_predictions
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class Resnet101V2_Test:
    
    MODEL_PATH = 'model_resnet101V2.h5'
    model_resnet101V2=load_model(MODEL_PATH)
    @staticmethod
    def model_resnet101V2_predict(img_path, model=model_resnet101V2):
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



