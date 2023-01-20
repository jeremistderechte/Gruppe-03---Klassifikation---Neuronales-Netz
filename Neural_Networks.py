# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 14:06:55 2023

@author: Jeremy Barenkamp
"""

import pandas as pd
import numpy as np
from collections import Counter
# =============================================================================
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
# =============================================================================
import tensorflow as tf

import re

import keras

from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


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


#Converting String to datetime
storno_cleaned_list = []
last_cleaned_list = []
for date_cancel, date_start in zip(df_data_new["cancel_date"][df_data_new["was_canceled"] == 1], 
                                   df_data_new["last_transaction"][df_data_new["was_canceled"] == 1],):
    date = re.sub(r'.[0-9][0-9][0-9]$', '', date_cancel)
    storno_cleaned_list.append(datetime.strptime(date, '%Y-%m-%d %H:%M:%S'))
    date = re.sub(r'.[0-9][0-9][0-9]$', '', date_start)
    last_cleaned_list.append(datetime.strptime(date, '%Y-%m-%d %H:%M:%S'))
    

df_dates = pd.DataFrame(columns=['last_date', 'cancel_date'])
df_dates['last_date'] = last_cleaned_list
df_dates['cancel_date'] = storno_cleaned_list

difference_list_year = []
differnce_list_month = []
differnce_list_day = []
for i, row in df_dates.iterrows():
    difference_list_year.append(abs(row['cancel_date'].year - row['last_date'].year))
    differnce_list_month.append(abs(row['cancel_date'].month - row['last_date'].month))
    differnce_list_day.append(abs(row['cancel_date'].day - row['last_date'].day))
    


#Drop obsolete columns
df_data_new = df_data_new.drop(["last_transaction", "cancel_date", "postal_code"], axis=1)

#Factorize input
df_data_new, string_factorize_sex, string_factorize_pay = factorize_label(df_data_new)

df_data_new = df_data_new.sample(frac=1)



#df_data_new.to_csv("cleanend.csv")

train, test = train_test_split(df_data_new, test_size=0.2, random_state=42)

print("Gefundene GPUs", tf.config.list_physical_devices('GPU'))

#Neural Network

#X-train

age = np.asarray(train["age"])

age = MinMaxScaler(feature_range=(0,1)).fit_transform(age.reshape(-1,1))

age = age.reshape(788,)

sex = np.asarray(train["sex"].astype("int32"))

payment_type = np.asarray(train["payment_type"].astype("int32"))

X = np.stack((age, sex, payment_type), axis=-1)

#X-Test

age_test = np.asarray(test["age"])

age_test = MinMaxScaler(feature_range=(0,1)).fit_transform(age_test.reshape(-1,1))

age_test = age_test.reshape(198,)

sex_test = np.asarray(test["sex"].astype("int32"))

payment_type_test = np.asarray(test["payment_type"].astype("int32"))
X_Test = np.stack((age_test, sex_test, payment_type_test), axis=-1)


#Y-Train
output = np.asarray(train["was_canceled"])

#Y-Test
y_test = (test['was_canceled']>0.1)





model = Sequential()
model.add(Dense(60, input_shape=(3,), activation='relu'))
model.add(Dense(30, activation='softmax'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


model.fit(X, output, batch_size=1, epochs=100)

test_predict = model.predict(X_Test)
test_predict =(test_predict>0.5)

y_test = (test['was_canceled']>0.1)





cm = confusion_matrix(y_test, test_predict, labels=[0,1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['nicht_storniert','storniert'])

disp.plot(cmap=plt.cm.Blues)

plt.show()




# =============================================================================
# input1 = keras.layers.Input(shape=(1,))
# input2 = keras.layers.Input(shape=(1,))
# input3 = keras.layers.Input(shape=(1,))
# merged = keras.layers.Concatenate(axis=1)([input1, input2, input3])
# dense1 = keras.layers.Dense(2, input_dim=3, activation=keras.activations.sigmoid, use_bias=True)(merged)
# output1 = keras.layers.Dense(1, activation=keras.activations.relu, use_bias=True)(dense1)
# model = keras.models.Model(inputs=[input1, input2, input3], outputs=output1)
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# =============================================================================


#test = np.stack(([23], [1], [2]), axis=-1)

#print(model.predict(test))

#model.fit([age, sex, payment_type],output, batch_size=4, epochs=10000)


