# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 07:27:44 2020

@author: Jagadheeswar Reddy
"""

from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions
import numpy as np
from tensorflow.keras.applications.vgg19 import VGG19
#from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class VGG19_Test:

   # MODEL_PATH = 'model_vgg19.h5'
   #model_vgg19=load_model(MODEL_PATH)
    model_vgg19= VGG19(
            include_top=True,
            weights="imagenet",
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
            classifier_activation="softmax",)
   

    @staticmethod
    def model_vgg19_predict(img_path, model=model_vgg19):
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



