#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 08:48:22 2023

@author: jeremy barenkamp
"""

#Demo for 'Introduction to data science'
from flask import Flask, render_template, request
from tensorflow import keras
import pickle
import numpy as np
import pandas as pd

model = keras.models.load_model('sonnenschein')
with open("encoder", "rb") as f: 
    encoder = pickle.load(f) 
    
with open("scaler", "rb") as f: 
    scaler = pickle.load(f) 

app = Flask(__name__)


def return_cancel_proba(age, sex, payment_type, threshold):
    
    
    #Conversion to numeric numbers
    
    if (sex == "male"):
        sex = 0
    else:
        sex = 1
        
    if (payment_type == "kreditkarte"):
        payment_type = 0
    elif (payment_type == "bar"):
        payment_type = 1
    elif (payment_type == "check"):
        payment_type = 2
        
    
    age = np.asarray(int(age))
    age = scaler.transform(age.reshape(-1,1))

    sex = int(sex)
    
    payment_type = np.asarray(int(payment_type))
    payment_type = encoder.transform(payment_type.reshape(1,1))

    
    df_input_data = pd.DataFrame(payment_type, columns = ['kreditkarte','bar','check'])
    df_input_data['sex'] = [sex]
    df_input_data['age'] = age

    #Let the model predict, if use cancels or not
    
    prediction = model.predict(df_input_data.to_numpy())
    print(prediction)
    
    if prediction > threshold:
        result = "Bitte Kunden Angebot schicken"
    else:
        result = "Dem Kunden muss kein Angebot gemacht werden"
        
    
    return result


@app.route('/', methods=["GET", "POST"])
def start():
    if request.method == "POST":
        age = request.form["age"]
        payment_type = request.form["pay_type"]
        gender = request.form["gender"]
        if request.form.get("prediction_security") == "cautious":
            threshold = 0.5
        elif request.form.get("prediction_security") == "risky":
            threshold = 0.63
        cancel_prediction = return_cancel_proba(age, gender, payment_type, threshold)
        return render_template('result.html', description=cancel_prediction)
        
    return render_template('demo.html')
