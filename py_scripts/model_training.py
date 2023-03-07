#Given a model and a set of parameters, train the model and return the results
#Run the model multiple times and average the results
import pandas as pd
import os
import sys
from torch import nn
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from data import get_training_data
import ner_util.ner_system as ner_util

def run_model(params, model, tokenizer, precentage=100, lang='swe',runs=5,uncased=False):
    #Get the data (X_train, Y_train, X_val, Y_val, X_test, Y_test)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = get_training_data(precentage,lang,uncased)
  
    #results
    results = pd.DataFrame([])

    #Fit the model
    for i in range(runs):
        print("Run: ", i)

        #Update the seed
        params.random_seed = i

        #Copy the model
        curr_model = model.copy()

        #Create a new instance of the model
        ner_system = ner_util.SequenceLabeler(params, curr_model, bert_tokenizer=tokenizer)
        
        #Fit the model
        ner_system.fit(X_train, Y_train, X_val, Y_val)

        #Get the evaluation results
        result = ner_system.evaluate_model(X_test,Y_test)

        #Add the results to the list
        results = results.append(result, ignore_index=True)

    
    #Average the results over each run
    results = results.mean(axis=0)
    return results