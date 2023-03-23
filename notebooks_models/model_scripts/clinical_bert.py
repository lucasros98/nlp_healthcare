import os
import sys
import pandas as pd
from torch import nn
from dotenv import load_dotenv, find_dotenv
sys.path.append(os.path.dirname(find_dotenv()))
load_dotenv(find_dotenv())

from py_scripts.model_training import run_model

#Get the precentage from the sys args
try:
    precentage = float(float(sys.argv[1])) if len(sys.argv) > 1 and sys.argv[1] != "None" else 100
except:
    precentage = 100
    print("Error occured while parsing the precentage from the sys args. Please check the sys args. Using {}% of the data.".format(precentage))


#Run the model
run_model(model_name="clinical_bert",bert_model="emilyalsentzer/Bio_ClinicalBERT",precentage=precentage,lang='en',runs=5)