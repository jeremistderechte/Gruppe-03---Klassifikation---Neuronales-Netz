#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 08:48:22 2023

@author: jeremy
"""

#Demo for 'Introduction to data science'
from flask import Flask, render_template
from tensorflow import keras
import numpy as np


app = Flask(__name__)


def return_cancel_proba(age, sex, payment_type):
    model = keras.models.load_model('sonnenschein')
    
    #Conversion to numeric numbers
    
    if (sex == "m"):
        sex = 1
    else:
        sex = 0
        
    if (payment_type == "Kreditkarte"):
        payment_type = 0
    elif (payment_type == "Bar"):
        payment_type = 1
    elif (payment_type == "Check"):
        payment_type = 2
        

    #merge all inputs to one list
    
    new_input = np.stack((age, sex, payment_type), axis=-1)
    
    #Let the model predict, if use cancels or not
    
    prediction = model.predict(new_input)
    
    threshold = 0.5
    
    if prediction > threshold:
        result = "Bitte Kunden Angebot schicken"
    else:
        result = "Kundem muss kein Angebot gemacht werden"
        
        
    return result
    
    



@app.route('/')
def start():
    return render_template('demo.html')