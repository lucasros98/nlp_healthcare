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
# For num_new_docs, we only want to compare one new document to its original document

data_size = 10
p_range = [None] # Use [None] for back_translation
aug_methods = ['back_translation'] #['unique_mention_replacement', 'local_mention_replacement', 'random_deletion', 'synonym_replacement', 'shuffle_within_segments', 'label_wise_token_replacement']
bt_type_range = ['w4w', 's4s'] # Use ['w4w', 's4s'] for back_translation, else [None]

#Get the training data
X_train, Y_train, _,_,_,_ = get_training_data(precentage=data_size)


# Calculate similarity scores for the different augmentation methods and parameter combinations
for aug_method in aug_methods:
    for p in p_range:
        for bt_type in bt_type_range:
            print(f"Calculating similarity scores for aug_method: {aug_method}, p: {p}, bt_type: {bt_type}")

            #Create the params for the augmentation
            params = {'aug_method': aug_method, 'p': p, 'num_new_docs': 1, 'data_size': data_size, 'bt_type': bt_type}
            
            #Get the augmented data
            X_aug, Y_aug = get_augmented_data(params=params)

            #Get the similarity scores between the original and the augmented data
            scores = compare_dataset(X_train, X_aug)

            #Save the scores
            df = pd.DataFrame(scores)

            # Calculate the mean over all documents for each score
            res = df.mean()

            #Save the results
            save_results(subfolder="similarity_10", filename=f"{aug_method}_{'p'+str(p) if p else ''}{bt_type if bt_type else ''}", result=res)

print("Similarity scores calculated and saved.")

        
        