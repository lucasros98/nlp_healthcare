#Explore the augmentation of the data and the differences for different data augmentation methods
# Path: nlp_healthcare/notebooks_exploration_cleaning/augmentation.py
import os
import sys
import pandas as pd
from torch import nn
from dotenv import load_dotenv, find_dotenv
sys.path.append(os.path.dirname(find_dotenv()))
load_dotenv(find_dotenv())

from py_scripts.file_handler import save_results
from py_scripts.data import get_training_data,get_augmented_data
from py_scripts.similarity import compare_dataset


#Data augmentation parameters
# We only want to generate one new document per document and compare it to the original document
p_range = [0.1, 0.3, 0.5]
aug_methods = ['random_deletion', 'synonym_replacement', 'shuffle_within_segments', 'label_wise_token_replacement', 'mention_replacement']

#Get the training data
X_train, Y_train, _,_,_,_ = get_training_data(precentage=100)


#Loop through the different augmentation methods and the different p values
for aug_method in aug_methods:
    for p in p_range:
        print(f"Running augmentation method: {aug_method}, p: {p}")
        #Create the params for the augmentation
        params = {'augmentation_type': aug_method, 'p': p, 'num_sentences': 1, 'data_size': 100}

        #Get the augmented data
        X_aug, Y_aug = get_augmented_data(params=params)

        #Get the similarity scores between the original and the augmented data
        scores = compare_dataset(X_train, X_aug)

        #Save the scores
        df = pd.DataFrame(scores)

        #Save the results
        save_results(subfolder="augmentation", filename=f"{aug_method}_p_{p}", result=df)


        
        