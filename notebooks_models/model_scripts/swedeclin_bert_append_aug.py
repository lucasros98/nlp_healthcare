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

combinations = [
    [
        {
            'aug_method': 'back_translation_s4s',
            'p': None,
            'num_new_docs': 1,
            'data_size': precentage
        },
        {
            'aug_method': 'bert_masking',
            'p': 0.1,
            'num_new_docs': 2,
            'data_size': precentage
        }
    ]
]

for aug_params_list in combinations:
    #Run the model
    run_model(model_name="swedeclin_bert",bert_model="/mimer/NOBACKUP/groups/snic2021-23-309/project-data/nlp-for-healthcare/SweDeClin-BERT_copy/",local_files_only=True,precentage=precentage,lang='sv',runs=5,aug_params_list=aug_params_list, aug_combination='append')
