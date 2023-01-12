# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 14:06:55 2023

@author: Jeremy Barenkamp
"""

import pandas as pd
import numpy as np
from collections import Counter

import tensorflow as tf

import keras

from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def factorize_label(dataframe):
    int_label_list_sex, string_factorize_sex = pd.factorize(dataframe["sex"])
    int_label_list_pay, string_factorize_pay = pd.factorize(dataframe["payment_type"])
    
    for i in range(len(dataframe.index)):
        dataframe.at[i, "sex"] = int_label_list_sex[i]
        dataframe.at[i, "payment_type"] = int_label_list_pay[i]
    
    
    return dataframe, string_factorize_sex, string_factorize_pay

#Adding column if payment was canceled
df_data = pd.read_csv("./python_datasets/VergangeneBestellungen.csv")

was_canceled = []
for i, row in df_data.iterrows():
    if str(row["Stornierungsdatum"]) == "nan":
        was_canceled.append(0)
    else:
        was_canceled.append(1)

df_data_new = pd.DataFrame(columns=["age", "sex", "postal_code", "payment_type", 
                                    "last_transaction", "cancel_date"
                                    , "was_canceled"])

df_data_new["age"] = df_data["Alter"]
df_data_new["sex"] = df_data["Geschlecht"]
df_data_new["postal_code"] = df_data["Postleitzahl"]
df_data_new["payment_type"] = df_data["Bezahlungsmethode"]
df_data_new["last_transaction"] = df_data["Letzte Transaktion"]
df_data_new["cancel_date"] = df_data["Stornierungsdatum"]
df_data_new["was_canceled"] = was_canceled

#Data Exploration

#maybe correlation
#print(Counter(df_data_new["age"][df_data_new["was_canceled"] == 1]))
#print(Counter(df_data_new["sex"][df_data_new["was_canceled"] == 1]))
#print(Counter(df_data_new["payment_type"][df_data_new["was_canceled"] == 1]))


#no correlation
#print(Counter(df_data_new["last_transaction"][df_data_new["was_canceled"] == 1]))
#print(Counter(df_data_new["postal_code"][df_data_new["was_canceled"] == 1])) 
#print(Counter(df_data_new["cancel_date"][df_data_new["was_canceled"] == 1]))

#Drop obsolete columns
df_data_new = df_data_new.drop(["last_transaction", "cancel_date", "postal_code"], axis=1)

#Factorize input
df_data_new, string_factorize_sex, string_factorize_pay = factorize_label(df_data_new)

df_data_new = df_data_new.sample(frac=1)

train, test = train_test_split(df_data_new, test_size=0.2, random_state=42)

print("Gefundene GPUs", tf.config.list_physical_devices('GPU'))

#Neural Network

#X

age = np.asarray(train["age"])

age = MinMaxScaler(feature_range=(0,1)).fit_transform(age.reshape(-1,1))

sex = np.asarray(train["sex"].astype("int32"))

payment_type = np.asarray(train["payment_type"].astype("int32"))


#Y
output = np.asarray(train["was_canceled"])


input1 = keras.layers.Input(shape=(1,))
input2 = keras.layers.Input(shape=(1,))
input3 = keras.layers.Input(shape=(1,))
merged = keras.layers.Concatenate(axis=1)([input1, input2, input3])
dense1 = keras.layers.Dense(2, input_dim=3, activation=keras.activations.sigmoid, use_bias=True)(merged)
output1 = keras.layers.Dense(1, activation=keras.activations.relu, use_bias=True)(dense1)
model = keras.models.Model(inputs=[input1, input2, input3], outputs=output1)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit([age, sex, payment_type],output, batch_size=16, epochs=100)


