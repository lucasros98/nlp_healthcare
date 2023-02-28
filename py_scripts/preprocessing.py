import string
import re
import os
import sys
from dotenv import load_dotenv,find_dotenv

sys.path.append(os.path.dirname(find_dotenv()))
load_dotenv(find_dotenv())

#Get the path for the data
PATH = os.getenv('DATA_PATH')
labels = ['First_Name', 'Last_Name', 'Phone_Number', 'Age', 'Full_Date', 'Date_Part', 'Health_Care_Unit', 'Location']

def remove_duplicates(X,Y):
    X_unique = []
    Y_unique = []

    for x,y in zip(X,Y):
        if x not in X_unique:
            X_unique.append(x)
            Y_unique.append(y)

    return X_unique,Y_unique

def clean_label_string(token,label_lower):
    token = token.replace('<' + label_lower + '>', '')
    token = token.replace('</' + label_lower + '>', '')
    return token

import string
import re

def preprocessing(IOB=True,punctuation=string.punctuation,no_duplicates=True,cased=False):
    #Load file and get lines
    with open(PATH) as f:
        documents = f.read().splitlines() 
    
    X = []
    Y = []

    #Create a dictionary to keep track of the number of inside labels
    nr_of_inside_labels = {}
    for label in labels:
        nr_of_inside_labels[label] = 0

    for doc in documents:
        curr_X = []
        curr_Y = []
        
        #to lowercase
        if(cased == False):
            doc = doc.lower()
        
        #add spaces between named entities
        doc = doc.replace(">","> ")
        doc = doc.replace("<"," <")

        #add spaces 
        doc = doc.replace("="," = ")
        doc = doc.replace("*"," * ")
        doc = doc.replace("+"," + ")
        
        doc = doc.replace("("," (")
        doc = doc.replace(")",") ")
        
        doc = doc.replace("->"," ->")

        #Add space after dot and comma when followed by a letter
        doc = re.sub(r'(?<=[.,:])(?=[a-zA-Z])', r' ', doc)
 
        #split string
        words = doc.split()
        
        #Skip empty lines
        if len(words) <= 2:
            continue
        
        named_entity = False
        inside_entity = False
        
        #loop over words, and mark each word as O or as their specific label
        for word in words:    

            #check if current token is a named entity
            if '<' in word[0] and '>' in word[-1]:
                
                #find the correct label
                for label in labels:
                    if cased == False:
                        label_lower = label.lower()
                    else:
                        label_lower = label

                    #start of entity
                    if '<' + label_lower + '>' in word:
                        word = clean_label_string(word,label_lower)
                        named_entity = True
                        break
                    #end of entity
                    elif '</' + label_lower + '>' in word:
                        word = clean_label_string(word,label_lower)
                        named_entity = False
                        inside_entity = False
                        break
                        
            #skip if empty string
            word = word.strip()
            word = word.strip(punctuation)
            
            #skip if empty string
            if len(word) == 0:
                continue
            
            #start of named entity    
            if named_entity and not inside_entity:
                if IOB:
                    curr_Y.append('B-'+label)
                else:
                    curr_Y.append(label)
                curr_X.append(word)
                inside_entity = True

            #inside of named entity
            elif named_entity and inside_entity:
                nr_of_inside_labels[label] += 1

                if IOB:
                    curr_Y.append('I-'+label)
                else:
                    curr_Y.append(label)
                curr_X.append(word)
        
            #outside of named enitity
            else:     
                curr_Y.append('O')
                curr_X.append(word)

        X.append(curr_X)
        Y.append(curr_Y)
    
    #Remove duplicates
    if no_duplicates:
        X,Y = remove_duplicates(X,Y)

    return X,Y