import pandas as pd
import os
import sys
from torch import nn
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import ner_util.ner_system as ner_util

def run_model(params, model, tokenizer, zipped, runs=5):
    #Get the data (X_train, Y_train, X_val, Y_val, X_test, Y_test)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = zipped
  
    #results
    results = []

    #Fit the model
    for i in range(runs):
        print("Run: ", i)

        #update the seed
        params["random_seed"] = i

        #Fit the model
        ner_system = ner_util.SequenceLabeler(params, model, bert_tokenizer=tokenizer)
        ner_system.fit(X_train, Y_train, X_val, Y_val)

        #Get the evaluation results
        result = ner_system.evaluate_model(X_test,Y_test)

        #Append the results
        results.append(result)
    
    #Average the results over each run
    results = pd.DataFrame(results)
    results = results.mean(axis=0)
    return results

    






