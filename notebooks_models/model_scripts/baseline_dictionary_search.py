#!/usr/bin/env python
# coding: utf-8

# # Baseline: Dictionary search
# 
# This notebooks explores another type of baseline for the system.
# 
# In this notebook a dictionary is created for the training data that maps all the named entities to its true label. 
# 
# The dictionary is then used for searching for the correct label, using the test data. If no corresponding class is found in the dictionary, the token is predicted as 'O'.

# ## Importing 

# In[ ]:


import os
import numpy as np
from dotenv import load_dotenv, find_dotenv
import sys

sys.path.append(os.path.dirname(find_dotenv()))
load_dotenv(find_dotenv())


# In[ ]:


#Import the file_handler.py file
from py_scripts.file_handler import save_result_file
from py_scripts.ner_util.ner_system import print_report
from py_scripts.data import get_training_data


# ## Creating dictionary from training data

# In[ ]:


#Create a dictionary/mapping of the labels from the training data
def create_mapping(X_train, Y_train,print_duplicates=False):
    mapping = {}

    for i in range(len(X_train)):
        for j in range(len(X_train[i])):
            if Y_train[i][j] != 'O':
                if X_train[i][j] in mapping and mapping[X_train[i][j]] != Y_train[i][j] and print_duplicates:
                    print("Duplicate word in training data: ", X_train[i][j], " with label: ", Y_train[i][j], " and ", mapping[X_train[i][j]])
                #map the word to the label
                mapping[X_train[i][j]] = Y_train[i][j]
    
    return mapping


# In[ ]:


#Predict the labels for the test data using the dictionary
def predict_labels(X_test, mapping):
    Y_pred = []

    for i in range(len(X_test)):
        Y_pred.append([])
        for j in range(len(X_test[i])):
            if X_test[i][j] in mapping:
                Y_pred[i].append(mapping[X_test[i][j]])
            else:
                Y_pred[i].append("O")
    
    return Y_pred


# ## Running the model

# In[ ]:


#Evaluate the model
import evaluate as ev
seqeval = ev.load('seqeval')

#Get the training data
data = [25,50,75,100]

for i in data:
    X_train, Y_train,_,_, X_test, Y_test = get_training_data(precentage=i)

    #Create the mapping
    mapping = create_mapping(X_train, Y_train)

    #Predict the labels
    Y_pred = predict_labels(X_test, mapping)

    #Evaluate the model
    print("Classification report for the model")

    results = seqeval.compute(predictions=Y_pred, references=Y_test, mode='strict', scheme='IOB2',zero_division=1)
    report = print_report(results)
    print(report)

    filename = "baseline_"+str(i)+".csv"
    save_result_file("baseline",filename, report)

