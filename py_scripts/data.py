import pandas as pd
import sys
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split

from dotenv import find_dotenv,load_dotenv
sys.path.append(os.path.dirname(find_dotenv()) + '/py_scripts')
from file_handler import read_public_csv,read_csv_file,write_csv_file
from generation import LabelGenerator
from preprocessing import remove_duplicates
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

def get_unique_test(lang="swe", uncased=True):
    """Get the unique test data.
    
    Returns:
        lists: The unique test data and labels. (X_test,Y_test)
    """
    if lang not in ['swe','eng']:
        print("Language need to be 'swe' or 'eng'")
        return

    #Check if we should use cased or uncased data
    if(uncased == True):
        name_test = "test_" + lang + "_unique.csv"
    else:
        name_test = "test_" + lang + "_cased_unique.csv"

    X_test,Y_test = read_csv_file(name_test,subfolder="test")
    return X_test,Y_test

def append_augmented_data(X, Y, params):
    """Append augmented data to training data sets X and Y.
    
    Args:
        X, Y (list): The lists of words and labels.
        params (dict): The augmentation parameters.

    Returns:
        lists: The new training data and labels. (X,Y)
    """

    _X = X.copy()
    _Y = Y.copy()
    augmentation_type, num_sentences, p, data_size = params
    file_name = augmentation_type + "_s" + str(num_sentences) + "_p" + str(p) + "_d" + str(data_size) + ".csv"

    print("Length of training data: " + str(len(_X)))
    X_aug, Y_aug = read_csv_file(file_name, subfolder="augmented")
    _X = _X + X_aug
    _Y = _Y + Y_aug
    print("Length of training data after augmentation type " + augmentation_type + ": " + str(len(_X)))
    _X, _Y = remove_duplicates(_X, _Y)
    print("Length of training data after removing duplicates: " + str(len(_X)))
    return _X, _Y

def get_training_data(precentage=100, lang='swe', uncased=True, unique_test=False):
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
    
    #Check if we should use cased or uncased data
    if(uncased == True):
        name_train = "train_" + lang + "_" + str(int(precentage)) + ".csv"
        name_val = "val_" + lang + ".csv"
        name_test = "test_" + lang + ("_unique" if unique_test else "") + ".csv"
    else:
        name_train = "train_" + lang + "_" + str(int(precentage)) + "_cased.csv"
        name_val = "val_" + lang + "_cased.csv"
        name_test = "test_" + lang + "_cased" + ("_unique" if unique_test else "") + ".csv"


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
    

    if not os.path.exists(data_path + 'train/cased'):
        os.makedirs(data_path + 'train/cased')
    
    
    if not os.path.exists(data_path + 'augmented'):
        os.makedirs(data_path + 'augmented')

    if not os.path.exists(data_path + 'val'):
        os.makedirs(data_path + 'val')
    
    if not os.path.exists(data_path + 'test'):
        os.makedirs(data_path + 'test')
    
    if not os.path.exists(data_path + 'processed'):
        os.makedirs(data_path + 'processed')

def generate_unique_test_data(uncased=True,lang="swe"):

    #Get the test and training data
    if lang == "swe":
        if uncased:
            X_train,Y_train = read_csv_file("train_swe_100.csv",subfolder="train")
            X_test,Y_test = read_csv_file("test_swe.csv",subfolder="test")
        else:
            X_train,Y_train = read_csv_file("train_swe_100_cased.csv",subfolder="train")
            X_test,Y_test = read_csv_file("test_swe_cased.csv",subfolder="test")
    elif lang == "eng":
        X_train,Y_train = read_csv_file("train_eng_100_cased.csv",subfolder="train")
        X_test,Y_test = read_csv_file("test_eng_cased.csv",subfolder="test")
    else:
        return

    #Create blacklist for entities from training data
    black_list = {}
    for i in range(len(Y_train)):
        doc = Y_train[i]
        inside_entity = False
        for j in range(len(doc)):
            #Check if start of entity
            if Y_train[i][j] == 'O':
                if inside_entity:
                    if entity in black_list:
                        black_list[entity].append(label_string)
                    else:
                        black_list[entity] = [label_string]
                    inside_entity = False
            
            elif Y_train[i][j][:2] == 'B-':
                #Check if we are already inside an entity -> then close the current entity
                if inside_entity:
                    if entity in black_list:
                        black_list[entity].append(label_string)
                    else:
                        black_list[entity] = [label_string]
                
                inside_entity = True
                entity = Y_train[i][j][2:]
                label_string = X_train[i][j]

            elif Y_train[i][j][:2] == 'I-':
                label_string = label_string + " " + X_train[i][j]
    
            #Check if end of document and we are inside an entity
            if j == len(doc)-1 and inside_entity:
                if entity in black_list:
                    black_list[entity].append(label_string)
                else:
                    black_list[entity] = [label_string]
                inside_entity = False
                label_string = ""

    #Create the label generator and set the blacklist
    label_gen = LabelGenerator()
    for key in black_list:
        black_list[key] = list(set(black_list[key]))
        label_gen.remove_common_entities(black_list[key],key)
    
    X_new = []
    Y_new = []
    #Update the test data to only have unique entities
    for i in range(len(Y_test)):
        doc = Y_test[i]
        inside_entity = False
        label_string = ""
        
        x_curr= []
        y_curr = []
        
        for j in range(len(doc)):
            #Check if start of entity (start with B-)
            if Y_test[i][j][:2] == 'B-':
                #Check if currently inside an entity -> then close the current entity
                if inside_entity and label_string in black_list[entity]:
                    #Generate new label
                    new_label = label_gen.generate_random_entity(entity)
                    new_label = str(new_label)
                    if new_label == None:
                        splitted = label_string.split(" ")

                    splitted = new_label.split(" ")
              
                    for k in range(len(splitted)):
                        if k == 0:
                            y_curr.append("B-" + entity)
                        else:
                            y_curr.append("I-" + entity)
                        
                        if uncased:
                            x_curr.append(splitted[k].lower())
                        else:
                            x_curr.append(splitted[k])
                    
                #Add the current entity to the blacklist
                inside_entity = True
                entity = Y_test[i][j][2:]
                label_string = X_test[i][j]

            elif Y_test[i][j][:2] == 'I-':
                label_string = label_string + " " + X_test[i][j]
            else:
                if inside_entity:
                    new_label = None
                    if label_string in black_list[entity]:
                        #Generate new label
                        new_label = label_gen.generate_random_entity(entity)
                        
                        #check if int -> then convert to string
                        if type(new_label) == int:
                            new_label = str(new_label)

                    if new_label == None:
                        splitted = label_string.split(" ")
                    else:
                        splitted = new_label.split(" ")
                    
                    for k in range(len(splitted)):
                        if k == 0:
                            y_curr.append("B-" + entity)
                        else:
                            y_curr.append("I-" + entity)
                        
                        if uncased:
                            x_curr.append(splitted[k].lower())
                        else:
                            x_curr.append(splitted[k])

                    inside_entity = False
                    label_string = ""

                y_curr.append(Y_test[i][j])
                x_curr.append(X_test[i][j])
            
            #Check if end of document and we are inside an entity
            if j == len(doc)-1 and inside_entity:
                if label_string in black_list[entity]:
                    #Generate new label
                    label_string = label_gen.generate_random_entity(entity)
                
                splitted = label_string.split(" ")
                for k in range(len(splitted)):
                    if k == 0:
                        y_curr.append("B-" + entity)
                    else:
                        y_curr.append("I-" + entity)
                    
                    if uncased:
                        x_curr.append(splitted[k].lower())
                    else:
                        x_curr.append(splitted[k])
                inside_entity = False
                label_string = ""

        X_new.append(x_curr)
        Y_new.append(y_curr)

                
    #Save the new test data
    if lang == "swe":
        if uncased:
            write_csv_file("test_swe_unique",X_new,Y_new,subfolder="test")
        else:
            write_csv_file("test_swe_cased_unique",X_new,Y_new,subfolder="test")

    elif lang == "eng":
            write_csv_file("test_eng_unique",X_new,Y_new,subfolder="test")

def split_data(X,Y,random_state=27):
    """Split the data into train, val, and test sets."""

    train_ratio = 0.80
    validation_ratio = 0.10
    test_ratio = 0.10

    #Split data into train, validation and test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1-train_ratio, random_state=random_state)
    X_validation, X_test, Y_validation, Y_test = train_test_split(X_test, Y_test, test_size=test_ratio/(test_ratio+validation_ratio), random_state=random_state)

    return X_train, Y_train, X_validation, Y_validation, X_test, Y_test

def count_labels(Y):
    """Count the labels in the data.
    """
    #Count the labels
    label_count = {}
    for doc in Y:
        for label in doc:
            if label[:2] == 'B-':
                if label[2:] in label_count:
                    label_count[label[2:]] += 1
                else:
                    label_count[label[2:]] = 1

    return label_count

def get_label_distribution(Y_full,Y_sub):
    """Get the distribution of labels in the full data and the sub data."""
    
    totalt_counts = count_labels(Y_full)
    sub_counts = count_labels(Y_sub)

    print("Label distribution:")
    for key in sub_counts:
        print(key + ": " + str(sub_counts[key] / totalt_counts[key]))

def split_randomly(X,Y,data_size=1.0,random_seed=27):
    if(data_size > 1 or data_size < 0):
        raise ValueError("Data size must be between 0 and 1")
    
    #Get the number of sentences
    num_sentences = len(X)

    #Get the number of sentences to use
    num_sentences_to_use = int(num_sentences * data_size)

    #Set the random seed
    random.seed(random_seed)

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