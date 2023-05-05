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

p_range = [0.1, 0.3, 0.5] # Use [None] for back_translation
num_new_docs_range = [1, 2, 3]
aug_methods =  ['unique_mention_replacement'] #['random_deletion', 'synonym_replacement', 'shuffle_within_segments', 'label_wise_token_replacement'] # ['back_translation']
bt_type = None

for aug_method in aug_methods:
    for p in p_range:
        for num_new_docs in num_new_docs_range:
            aug_params_list = [
                {
                    'aug_method': aug_method,
                    'p': p,
                    'num_new_docs': num_new_docs,
                    'data_size': precentage,
                    'bt_type': bt_type
                }
            ]
            #Run the model
            run_model(model_name="swedeclin_bert",bert_model="/mimer/NOBACKUP/groups/snic2021-23-309/project-data/nlp-for-healthcare/SweDeClin-BERT_copy/",local_files_only=True,precentage=precentage,lang='sv',runs=5,aug_params_list=aug_params_list)
