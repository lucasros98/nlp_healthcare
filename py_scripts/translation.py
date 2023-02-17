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

from file_handler import read_csv_file,write_csv_file

#Change to cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#if(device == "cuda"):
#    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:8"

print("Running on: ",device)
#This function translate Swedish text data into English by using 
# sense for sense translation
def translate_text_to_eng(X,Y):

    X_masked, mapping = mask_entities(X,Y)

    model_en.to(device)
    X_masked = tokenize_en(X_masked, return_tensors="pt",padding=True).to(device)

    X_translated = model_en.generate(**X_masked, num_beams=3, max_length=512, early_stopping=True).to("cpu")

    X_translated = tokenize_en.batch_decode(X_translated, skip_special_tokens=True)

    entities = []
    for key, value in mapping.items():
        entities.append(value[0])

    if(len(entities) != 0):
        entities = tokenize_en(entities,padding=True, return_tensors="pt").to(device)
        entities = model_en.generate(**entities, num_beams=3, max_length=512, early_stopping=True).to("cpu")
        entities = tokenize_en.batch_decode(entities, skip_special_tokens=True)
        
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

def translate_from_file(filename,batch_size=64):

    if(filename == None):
        return None,None            

    print("Reading file...")
    X,Y = read_csv_file(filename)

    X_res, Y_res = [],[]
    print("Starting to process batches...")
    for i in tqdm(range(0,len(X),batch_size)):
        X_batch = X[i:i+batch_size]
        Y_batch = Y[i:i+batch_size]

        #Translate the batch
        X_translated, Y_translated = translate_text_to_eng(X_batch,Y_batch)

        #Append the results
        X_res += X_translated
        Y_res += Y_translated

    #print first 10 results 
    print(X_res[:10])
    return X_res,Y_res
    

X,Y = translate_from_file("clean.csv")
write_csv_file("translated",X,Y)