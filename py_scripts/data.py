import pandas as pd
import sys
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split

from dotenv import find_dotenv,load_dotenv
sys.path.append(os.path.dirname(find_dotenv()) + '/py_scripts')
from file_handler import read_public_csv,read_csv_file
load_dotenv(find_dotenv())

def get_tokens(data):
    """
    Get the tokens in the data, and the number of times they appear.

    Args: 
        list: The data to check.

    Returns:
        dict: The tokens in the data, and the number of times they appear.
    """

    tokens = {}
    for sentence in data:
        for word in sentence:
            if word not in tokens:
                tokens[word] = 1
            else:
                tokens[word] += 1
    return tokens

def get_unknown_tokens(tokenizer, data):
    """Get the unknown tokens in the data.

    Args:
        tokenizer (Tokenizer): The tokenizer to use.
        data (list): The data to check.
    Returns:
        list: The unknown tokens in the data.
    """
    tokens = get_tokens(data)
    unknown_tokens = []

    for token in tokens:
        #Tokenize the token
        tokenized = tokenizer([token], is_split_into_words=True, truncation=True, max_length=512)
        
        input_ids = tokenized.input_ids

        s = tokenizer.decode(input_ids)

        s = s.replace("[SEP]", "")
        s = s.replace("[CLS]", "")

        if "[UNK]" in s:
            unknown_tokens.append({"token": token, "count": tokens[token], "tokenized": s})

    return unknown_tokens 

#Print the n most common unknown tokens in the data      
def print_unknown_tokens(tokenizer, data, n=30):

    unknown_tokens = get_unknown_tokens(tokenizer, data)

    df = pd.DataFrame(unknown_tokens)
    
    if df.empty:
        print('No UNK tokens found')
    else:
        df = df.sort_values(by=['count'], ascending=False)
        df = df.head(n)
        print(df)

def get_training_data(precentage=100,lang='swe'):
    """Get the training data. Precentage need to be 25, 50, 75 or 100.
    
    Args:
        precentage (int): The precentage of the training data to use.
        lang (str): The language to use. (swe or eng)

    Returns:
        lists: The training data and labels. (X_train,Y_train,X_val,Y_val,X_test,Y_test)
    """

    if precentage not in [25,50,75,100]:
        print("Precentage need to be 25, 50, 75 or 100")
        return

    if lang not in ['swe','eng']:
        print("Language need to be 'swe' or 'eng'")
        return

    name_train = "train_" + lang + "_" + str(int(precentage)) + ".csv"
    name_val = "val_" + lang + ".csv"
    name_test = "test_" + lang + ".csv"

    X_train,Y_train = read_csv_file(name_train,subfolder="train")
    X_val,Y_val = read_csv_file(name_val,subfolder="val")
    X_test,Y_test = read_csv_file(name_test,subfolder="test")

    return X_train,Y_train,X_val,Y_val,X_test,Y_test

def create_data_dirs():
    """Create the data directories. (val, test, train, augmented, processed)
    """
    #check if the DATA_DIR environment variable is set
    if os.environ.get("DATA_DIR") == None:
        print("Please set the DATA_DIR environment variable.")
        return

    data_path = os.environ.get("DATA_DIR")

    #Create the data directories
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    if not os.path.exists(data_path + 'train'):
        os.makedirs(data_path + 'train')
    
    if not os.path.exists(data_path + 'augmented'):
        os.makedirs(data_path + 'augmented')

    if not os.path.exists(data_path + 'val'):
        os.makedirs(data_path + 'val')
    
    if not os.path.exists(data_path + 'test'):
        os.makedirs(data_path + 'test')
    
    if not os.path.exists(data_path + 'processed'):
        os.makedirs(data_path + 'processed')


def split_data(X,Y,random_state=27):
    """Split the data into train, val, and test sets."""

    train_ratio = 0.80
    validation_ratio = 0.10
    test_ratio = 0.10

    #Split data into train, validation and test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1-train_ratio, random_state=random_state)
    X_validation, X_test, Y_validation, Y_test = train_test_split(X_test, Y_test, test_size=test_ratio/(test_ratio+validation_ratio), random_state=random_state)

    return X_train, Y_train, X_validation, Y_validation, X_test, Y_test

def split_randomly(X,Y,data_size=1.0):
    if(data_size > 1 or data_size < 0):
        raise ValueError("Data size must be between 0 and 1")
    
    #Get the number of sentences
    num_sentences = len(X)

    #Get the number of sentences to use
    num_sentences_to_use = int(num_sentences * data_size)

    #Get the indices of the sentences to use
    indices = random.sample(range(num_sentences), num_sentences_to_use)

    #Get the sentences to use
    X_new = []
    Y_new = []
    for i in range(len(indices)):
        X_new.append(X[indices[i]])
        Y_new.append(Y[indices[i]])

    return X_new, Y_new


#Replace the abbreviations in the data with the corresponding tokens
def decode_abbrevs(X,Y):
    dict = read_public_csv("abbreviations.csv")

    X_new, Y_new = [],[]

    for i in range(len(X)):
        curr_x, curr_y = [],[]

        for j in range(len(X[i])):
            if(X[i][j] in dict):
                #Get the abbreviation
                abrev = dict[X[i][j]]

                #Tokenize the abbreviation
                abrev = abrev.split(" ")

                #Replace the abbreviation with the tokens
                for k in range(len(abrev)):
                    curr_x.append(abrev[k])
                    curr_y.append(Y[i][j])
            else:
                #Append the original word
                curr_x.append(X[i][j])
                curr_y.append(Y[i][j])
        X_new.append(curr_x)
        Y_new.append(curr_y)

    return X_new, Y_new