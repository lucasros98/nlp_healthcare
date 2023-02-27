import pandas as pd
import sys
import os
import random
import numpy as np

from dotenv import find_dotenv
sys.path.append(os.path.dirname(find_dotenv()) + '/py_scripts')
from file_handler import read_public_csv

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

#The input is a list of lists, where each list is a sentence
#data size is the percentage of the data to use
#the data is split randomly
def split_randomly(X,Y,data_size=1):
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