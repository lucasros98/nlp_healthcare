#To translate text data from Swedish to English and vice versa
#We use the MarianMT model from HuggingFace
from file_handler import read_csv_file
from transformers import MarianMTModel, MarianTokenizer
import string

#Load MarianMT models from HuggingFace
model_name_swe_to_en = 'Helsinki-NLP/opus-mt-sv-en'
model_en = MarianMTModel.from_pretrained(model_name_swe_to_en)
tokenize_en = MarianTokenizer.from_pretrained(model_name_swe_to_en)

model_name_en_to_swe = 'Helsinki-NLP/opus-mt-en-sv'
model_swe = MarianMTModel.from_pretrained(model_name_en_to_swe)
tokenize_swe = MarianTokenizer.from_pretrained(model_name_en_to_swe)

#This function translate Swedish text data into English by using 
# sense for sense translation
def translate_text_to_eng(X,Y):

    #mask entities
    X_masked, mapping = mask_entities(X,Y)

    #tokenize the text
    X_masked = [tokenize_en.encode(t, return_tensors="pt") for t in X_masked]

    #translate sense into English
    X_translated = [model_en.generate(t, num_beams=4, max_length=400) for t in X_masked]

    #decode the translation
    X_translated = [tokenize_en.decode(t[0], skip_special_tokens=True) for t in X_translated]

    #translate mapping to English
    for key, value in mapping.items():
        entity = value[0]
        entity = tokenize_en.encode(entity, return_tensors="pt")
        entity = model_en.generate(entity, num_beams=4, max_length=400, early_stopping=True)
        entity = tokenize_en.decode(entity[0], skip_special_tokens=True)
        mapping[key] = [entity, value[1]]
    
    #Insert the entities back into the text
    #also create new Y for the translated text
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

    #return the translated text X and the new Y
    return new_X, new_Y


def translate_text_to_swe(X,Y):
    
    #mask entities
    X_masked, mapping = mask_entities(X,Y)

    #tokenize the text
    X_masked = [tokenize_swe.encode(t, return_tensors="pt") for t in X_masked]

    #translate sense into Swedish
    X_translated = [model_swe.generate(t, num_beams=4, max_length=400) for t in X_masked]

    #decode the translation
    X_translated = [tokenize_swe.decode(t[0], skip_special_tokens=True) for t in X_translated]

    #translate mapping to Swedish
    for key, value in mapping.items():
        entity = value[0]
        entity = tokenize_swe.encode(entity, return_tensors="pt")
        entity = model_swe.generate(entity, num_beams=4, max_length=400, early_stopping=True)
        entity = tokenize_swe.decode(entity[0], skip_special_tokens=True)
        mapping[key] = [entity, value[1]]
    
    #Insert the entities back into the text
    #also create new Y for the translated text
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
    
    #return the translated text X and the new Y
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
        return translate_text_to_eng(X,Y)
    else:
        X,Y = read_csv_file(filename)
        return translate_text_to_swe(X,Y)