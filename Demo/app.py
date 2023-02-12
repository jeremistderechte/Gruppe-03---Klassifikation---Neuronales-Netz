#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 08:48:22 2023

@author: jeremy barenkamp
"""

#Demo for 'Introduction to data science'
from flask import Flask, render_template, request
from tensorflow import keras

app = Flask(__name__)


def return_cancel_proba(age, sex, payment_type, threshold):
    model = keras.models.load_model('sonnenschein')
    
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
        
    
    age = int(age)
    sex= int(sex)
    payment_type = int(payment_type)

    #Let the model predict, if use cancels or not
    
    prediction = model.predict([[age, sex, payment_type]])
    print(prediction)
    
    if prediction > threshold:
        result = "Bitte Kunden Angebot schicken"
    else:
        result = "Kundem muss kein Angebot gemacht werden"
        
    
    return result


@app.route('/', methods=["GET", "POST"])
def start():
    if request.method == "POST":
        age = request.form["age"]
        payment_type = request.form["pay_type"]
        gender = request.form["gender"]
        if request.form.get("prediction_security") == "cautious":
            threshold = 0.7
        elif request.form.get("prediction_security") == "risky":
            threshold = 0.5
        cancel_prediction = return_cancel_proba(age, gender, payment_type, threshold)
        return render_template('result.html', description=cancel_prediction)
        
    return render_template('demo.html')
