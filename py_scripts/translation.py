from transformers import MarianMTModel, MarianTokenizer
import string
import sys
from tqdm import tqdm
import torch
import os

#Load MarianMT models from HuggingFace
model_name_swe_to_en = 'Helsinki-NLP/opus-mt-sv-en'
model_en = MarianMTModel.from_pretrained(model_name_swe_to_en)
tokenize_en = MarianTokenizer.from_pretrained(model_name_swe_to_en)

model_name_en_to_swe = 'Helsinki-NLP/opus-mt-en-sv'
model_swe = MarianMTModel.from_pretrained(model_name_en_to_swe)
tokenize_swe = MarianTokenizer.from_pretrained(model_name_en_to_swe)

#Base import on the path when importing from another file
#The path will need to be nlp_healthcare/py_scripts
from dotenv import find_dotenv,load_dotenv
sys.path.append(os.path.dirname(find_dotenv()) + '/py_scripts')
load_dotenv(find_dotenv())

from file_handler import read_csv_file

#Change to cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#This function translate Swedish text data into English by using 
# sense for sense translation
def translate_text_to_eng(X,Y):

    #mask entities
    print("Masking entities...")
    X_masked, mapping = mask_entities(X,Y)

    #Tokenize the text
    print("Tokenizing the text...")
    X_masked = tokenize_en(X_masked, return_tensors="pt",padding=True)

    model_en.to(device)
    X_masked = X_masked.to(device)

    #translate sense into English
    print("Translating the text...")
    X_translated = model_en.generate(**X_masked, num_beams=2, max_length=512, early_stopping=True)

    #decode the translation
    print("Decoding the text...")
    X_translated = tokenize_en.batch_decode(X_translated, skip_special_tokens=True)

    #translate mapping to English
    print("Translating the mapping...")
    entities = [mapping[i][0] for i in range(len(mapping))]
    entities = tokenize_en(entities,padding=True, return_tensors="pt")
    entities = entities.to(device)
    entities = model_en.generate(**entities, num_beams=2, max_length=512, early_stopping=True)
    entities = tokenize_en.batch_decode(entities, skip_special_tokens=True)
    for i in range(len(mapping)):
        mapping[i][0] = entities[i]

    #Insert the entities back into the text
    mapping = mapping.to("cpu")
    X_translated = X_translated.to("cpu")
    
    #Insert the entities back into the text
    #also create new Y for the translated text
    X_new, Y_new = unmask_entities(X_translated, mapping)
    
    #return the translated text X and the new Y
    print("Returning the translated text...")
    print(X_new[0], Y_new[0])


def translate_text_to_swe(X,Y):
    
    #mask entities
    print("Masking entities...")
    X_masked, mapping = mask_entities(X,Y)

    #Change X-masked to current device
    X_masked = X_masked.to(device)
    mapping = mapping.to(device)

    #tokenize the text
    print("Tokenizing the text...")
    X_masked = tokenize_swe(X_masked, return_tensors="pt")

    #translate sense into Swedish
    print("Translating the text...")
    X_translated = model_swe.generate(**X_masked, num_beams=3, max_length=512, early_stopping=True)

    #decode the translation
    print("Decoding the text...")
    X_translated = tokenize_swe.batch_decode(X_translated, skip_special_tokens=True)

    #translate mapping to Swedish
    print("Translating the mapping...")
    entities = [mapping[i][0] for i in range(len(mapping))]
    entities = tokenize_en(entities, return_tensors="pt")
    entities = model_en.generate(**entities, num_beams=2, max_length=512, early_stopping=True)
    entities = tokenize_en.batch_decode(entities, skip_special_tokens=True)
    for i in range(len(mapping)):
        mapping[i][0] = entities[i]

    #change to cpu
    mapping = mapping.to("cpu")
    X_translated = X_translated.to("cpu")
    
    #Insert the entities back into the text
    #also create new Y for the translated text
    X_new, Y_new = unmask_entities(X_translated, mapping)
    
    #return the translated text X and the new Y
    return X_new, Y_new

def unmask_entities(X_translated, mapping):
    new_X = []
    new_Y = []

    for i in range(len(X_translated)):
        #tokenize the text
        x = X_translated[i].split(" ")

        curr_y = []
        curr_x = []

        for j in range(len(x)):
            #remove punctuation
            word = x[j].strip(string.punctuation)
   
            #Check if the word could be an entity
            if(word[0] == "X"):
                #Check if the word is an entity
                if(word in mapping):
                    curr_y.append(mapping[word][1])
                    curr_x.append(mapping[word][0])
                else:
                    curr_y.append("O")
                    curr_x.append(word)
            else:
                curr_y.append("O")
                curr_x.append(word)

        new_X.append(curr_x)
        new_Y.append(curr_y)
    return new_X, new_Y
    

def mask_entities(X,Y):
    new_X = []
    linkage = {}
    for i in range(len(Y)):
        curr_y=Y[i]
        curr_x=X[i]

        for j in range(len(curr_y)):
            if(curr_y[j] != "O"):
                
                #Create dummy word
                dummy = "X"+str(i)+"-"+str(j)
                linkage[dummy] = [curr_x[j],curr_y[j]]
                curr_x[j] = dummy
        new_X.append(curr_x)
    
    for x in new_X:
        new_string = ""
        for y in range(len(x)):
            new_string += x[y]
            if(y != len(x)-1):
                new_string += " "
        new_X[new_X.index(x)] = new_string
    
    return new_X, linkage

def translate_from_file(filename, target="en"):
    if target == "en":
        X,Y = read_csv_file(filename)
        print(X[0])
        return translate_text_to_eng(X,Y)
    elif target == "sv":
        X,Y = read_csv_file(filename)
        return translate_text_to_swe(X,Y)
    else:
        print("Target language not supported")
        return None
    


translate_from_file("clean.csv", target="en")