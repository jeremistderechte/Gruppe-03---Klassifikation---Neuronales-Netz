# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 14:06:55 2023

@author: Jeremy Barenkamp
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
'''
Runs on CPU even if compatible gpu is installed, because gpu is slower in this case
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
import pickle
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score




################## ################## ##################




#Factorization of Label (means "string to int")
def factorize_label(dataframe):
    int_label_list_sex, string_factorize_sex = pd.factorize(dataframe["sex"])
    int_label_list_pay, string_factorize_pay = pd.factorize(dataframe["payment_type"])
    for i in range(len(dataframe.index)):
        dataframe.at[i, "sex"] = int_label_list_sex[i]
        dataframe.at[i, "payment_type"] = int_label_list_pay[i]
    
    return dataframe, string_factorize_sex, string_factorize_pay

#Evaluates model with precision, recall and accuracy
def evaluate_model(trained_model, test_data_x, test_data_y):
    test_predict_y = trained_model.predict(test_data_x)
    test_predict_y = (test_predict_y>0.5)
    
    precision = precision_score(test_data_y, test_predict_y)
    recall = recall_score(test_data_y, test_predict_y)
    accuracy = accuracy_score(test_data_y, test_predict_y)
    
    return precision, recall, accuracy

#data preprocessing

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

#Transforming date string to datetime type
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


#Saving cleaned data
#df_data_new.to_csv("cleanend.csv")

#Train-Test-Split

train, test = train_test_split(df_data_new, test_size=0.2, random_state=42)

#test.to_csv("./test_data.csv")

print("Gefundene GPUs", tf.config.list_physical_devices('GPU'))

#Neural Network

onehot_encoder = OneHotEncoder(sparse=False)

scaler = MinMaxScaler(feature_range=(0,1))

#X-train

age = np.asarray(train["age"])

age = scaler.fit_transform(age.reshape(-1,1))

age = age.reshape(788,)

sex = np.asarray(train["sex"].astype("int32"))

payment_type = np.asarray(train["payment_type"].astype("int32"))

payment_type = payment_type.reshape(len(payment_type), 1)


payment_type = onehot_encoder.fit_transform(payment_type)

df_hot_encoded = pd.DataFrame(payment_type, columns = ['kreditkarte','bar','check'])
df_hot_encoded["sex"] = sex
df_hot_encoded["age"] = age

X = df_hot_encoded.to_numpy()




#X = np.stack((age, sex, payment_type), axis=-1)

#X-Test

age_test = np.asarray(test["age"])

age_test = scaler.transform(age_test.reshape(-1,1))

age_test = age_test.reshape(198,)

sex_test = np.asarray(test["sex"].astype("int32"))


payment_type_test = np.asarray(test["payment_type"].astype("int32"))

payment_type_test = payment_type_test.reshape(len(payment_type_test), 1)

payment_type_test = onehot_encoder.transform(payment_type_test)

df_hot_encoded_test = pd.DataFrame(payment_type_test, columns = ['kreditkarte','bar','check'])
df_hot_encoded_test["sex"] = sex_test
df_hot_encoded_test["age"] = age_test

X_Test = df_hot_encoded_test.to_numpy()

df_hot_encoded_test.to_csv("test_data_hot_encoded.csv")

# =============================================================================
# with open("encoder", "wb") as f: 
#     pickle.dump(onehot_encoder, f)
#     
# with open("scaler", "wb") as f: 
#     pickle.dump(scaler, f)
# =============================================================================


#X_Test = np.stack((age_test, sex_test, payment_type_test), axis=-1)


#Y-Train
output = np.asarray(train["was_canceled"])

#Y-Test
y_test = (test['was_canceled']>0.1)

#y_test.to_csv("y_test.csv")


# Best result: Neuron = 128 batchsize = 8 epochs = 10


# Write comment into list to test best parameters

neuron_list = [128] #512, 256, 128, 64, 32, 16, 8, 4

batch_size_list = [8]# 1, 2, 4, 8, 16, 32, 64, 128, 256

epoch_list = [10] # 5, 10, 30, 90, 180



precision_list = []
recall_list = []
accuracy_list = []
parameter_list = []



df_results_neural_network = pd.DataFrame(columns=["parameter", "precision", "recall", "accuracy"])


# Neural Network
    

for epoch in epoch_list:

    #print(model.summary())
    
    for batch_size in batch_size_list:
        for neuron in neuron_list:
            
            model = Sequential()
            model.add(Dense(neuron, input_shape=(5,), activation='relu'))
            model.add(Dense(neuron/2, activation='relu'))
            model.add(Dense(1, activation="sigmoid"))
            
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            
            #callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)]
            
            model.fit(X, output, batch_size=batch_size, epochs=epoch, validation_split=0.1, shuffle=True,
                      )
            
            print("Neurons:", neuron, "Batch_Size:", batch_size,"Epochs:", epoch, "\n")
            
            
            
            precision, recall, accuracy = evaluate_model(model, X_Test, (test['was_canceled']>0.5))
            
            
            precision_list.append(precision)
            recall_list.append(recall)
            accuracy_list.append(accuracy)
            
                        
            parameter_list.append([neuron, batch_size, epoch])
        
            
            
df_results_neural_network["precision"] =  precision_list
df_results_neural_network["recall"] =  recall_list
df_results_neural_network["accuracy"] =  accuracy_list
df_results_neural_network["parameter"] = parameter_list


test_predict = model.predict(X_Test)
test_predict =(test_predict>0.5)

y_test = (test['was_canceled']>0.5)

# Just if one model is trained
if (len(neuron_list) == 1 and len(batch_size_list) == 1 and len(epoch_list)) == 1:

    # Ploting confusion matrix
    
    cm = confusion_matrix(y_test, test_predict, labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['nicht_storniert','storniert'])
    
    disp.plot(cmap=plt.cm.Blues)
    
    plt.show()
    
    # Saving model
    #model.save("./models/sonnenschein")





#Data Exploration

#maybe correlation
#print(Counter(df_data_new["age"][df_data_new["was_canceled"] == 1]))
#print(Counter(df_data_new["sex"][df_data_new["was_canceled"] == 1]))
#print(Counter(df_data_new["payment_type"][df_data_new["was_canceled"] == 1]))


#no correlation
#print(Counter(df_data_new["last_transaction"][df_data_new["was_canceled"] == 1]))
#print(Counter(df_data_new["postal_code"][df_data_new["was_canceled"] == 1])) 
#print(Counter(df_data_new["cancel_date"][df_data_new["was_canceled"] == 1]))
