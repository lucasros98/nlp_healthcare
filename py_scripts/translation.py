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
from dotenv import find_dotenv,load_dotenv
sys.path.append(os.path.dirname(find_dotenv()) + '/py_scripts')
load_dotenv(find_dotenv())

from file_handler import read_csv_file, write_csv_file
from data import decode_abbrevs

#Change to cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on: ",device)

#Parameters for translation models
class TranslationParameters():
    num_beams=4
    early_stopping=True
    max_length=512
    use_decoded=False

#Translate Swedish text data into English by using 
def translate_text_to_eng(X,Y,params=TranslationParameters()):

    X_masked, mapping = mask_entities(X,Y)

    model_en.to(device)
    X_masked = tokenize_en(X_masked, return_tensors="pt",padding=True).to(device)

    X_translated = model_en.generate(**X_masked, num_beams=params.num_beams, max_length=params.max_length, early_stopping=params.early_stopping).to("cpu")

    X_translated = tokenize_en.batch_decode(X_translated, skip_special_tokens=True)

    entities = []
    for key, value in mapping.items():
        entities.append(value[0])

    if(len(entities) != 0):
        entities = tokenize_en(entities, padding=True, return_tensors="pt").to(device)
        entities = model_en.generate(**entities, num_beams=params.num_beams, max_length=params.max_length, early_stopping=params.early_stopping).to("cpu")
        entities = tokenize_en.batch_decode(entities, skip_special_tokens=True)
        
        #Append the entities to the mapping
        for i in range(len(entities)):
            mapping[list(mapping.keys())[i]].append(entities[i])

    #Insert the entities back into the text
    #also create new Y for the translated text
    X_new, Y_new = unmask_entities(X_translated, mapping)
    
    return X_new, Y_new

#Translate Swedish text data into English by using 
def translate_text_to_swe(X,Y,params=TranslationParameters()):

    X_masked, mapping = mask_entities(X,Y)

    model_swe.to(device)
    X_masked = tokenize_swe(X_masked, return_tensors="pt",padding=True).to(device)

    X_translated = model_swe.generate(**X_masked, num_beams=params.num_beams, max_length=params.max_length, early_stopping=params.early_stopping).to("cpu")

    X_translated = tokenize_swe.batch_decode(X_translated, skip_special_tokens=True)

    entities = []
    for key, value in mapping.items():
        entities.append(value[0])

    if(len(entities) != 0):
        entities = tokenize_swe(entities, padding=True, return_tensors="pt").to(device)
        entities = model_swe.generate(**entities, num_beams=params.num_beams, max_length=params.max_length, early_stopping=params.early_stopping).to("cpu")
        entities = tokenize_swe.batch_decode(entities, skip_special_tokens=True)
        
        #Append the entities to the mapping
        for i in range(len(entities)):
            mapping[list(mapping.keys())[i]].append(entities[i])

    #Insert the entities back into the text
    #also create new Y for the translated text
    X_new, Y_new = unmask_entities(X_translated, mapping)
    
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
            
            #continue if the word is empty
            if(len(word) == 0):
                continue
    
            #Check if the word could be an entity
            if(word[0] == "X"):
                #Check if the word is an entity
                if(word in mapping):
                    curr_y.append(mapping[word][1])
                    curr_x.append(mapping[word][0])
                else:
                    special_case = False
                    #check if a substring of the word is an entity
                    #this is a special case for the translation model
                    for key, value in mapping.items():
                        if(key in word):
                            special_case = True
                            #replace the substring with the entity
                            new_word = word.replace(key,value[0])
                            curr_y.append(value[1])
                            curr_x.append(new_word)
                            break
                    if not special_case:
                        curr_y.append("O")
                        curr_x.append(word)
            else:
                curr_y.append("O")
                curr_x.append(word)

        new_X.append(curr_x)
        new_Y.append(curr_y)
    return new_X, new_Y
    

def mask_entities(X_data,Y_data):
    X = X_data.copy()
    Y = Y_data.copy()

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

def translate_text_to_eng_batch(X,Y,params=TranslationParameters(),batch_size=64):
    #Decode clincal abbreviations
    if params.use_decoded:
       X,Y = decode_abbrevs(X,Y)

    X_res, Y_res = [],[]
    print("Starting to process batches...")
    for i in tqdm(range(0,len(X),batch_size)):
        X_batch = X[i:i+batch_size]
        Y_batch = Y[i:i+batch_size]

        #Translate the batch
        X_translated, Y_translated = translate_text_to_eng(X_batch,Y_batch,params=params)

        #Append the results
        X_res += X_translated
        Y_res += Y_translated

    #clear the cache
    torch.cuda.empty_cache()

    return X_res,Y_res

def translate_text_to_swe_batch(X,Y,params=TranslationParameters(),batch_size=64):
    #Decode clincal abbreviations

    X_res, Y_res = [],[]
    print("Starting to process batches...")
    for i in tqdm(range(0,len(X),batch_size)):
        X_batch = X[i:i+batch_size]
        Y_batch = Y[i:i+batch_size]

        #Translate the batch
        X_translated, Y_translated = translate_text_to_swe(X_batch,Y_batch,params=params)

        #Append the results
        X_res += X_translated
        Y_res += Y_translated
    
    #clear the cache
    torch.cuda.empty_cache()

    return X_res,Y_res


def translate_from_file(filename,batch_size=64):

    params = TranslationParameters()

    if(filename == None):
        return None,None            

    print("Reading file...")
    X,Y = read_csv_file(filename)

    #Decode clincal abbreviations
    if params.use_decoded:
       X,Y = decode_abbrevs(X,Y)

    X_res, Y_res = [],[]
    print("Starting to process batches...")
    for i in tqdm(range(0,len(X),batch_size)):
        X_batch = X[i:i+batch_size]
        Y_batch = Y[i:i+batch_size]

        #Translate the batch
        X_translated, Y_translated = translate_text_to_eng(X_batch,Y_batch,params=params)

        #Append the results
        X_res += X_translated
        Y_res += Y_translated

    #print first 10 results 
    return X_res,Y_res