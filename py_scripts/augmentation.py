# Functions for data augmentation
import numpy as np
from numpy import random
#from translation import translate_text_to_eng_batch, translate_text_to_swe_batch
from file_handler import read_csv_file, write_csv_file
from generation import LabelGenerator
from tqdm import tqdm

binomial_p = 0.7
num_new_docs = 3

def data_augmentation(X, Y, aug_type):
    X_new = X.copy()
    Y_new = Y.copy()

    for sentence, labels in tqdm(zip(X,Y), desc=f'aug_type={aug_type} with p={binomial_p}, num_new_docs={num_new_docs}'):
        for i in range(num_new_docs):
            if aug_type == "synonym_replacement":
                sentence_new, labels_new = synonym_replacement(sentence, labels)
                X_new.append(sentence_new)
                Y_new.append(labels_new)
            elif aug_type == "random_deletion":
                sentence_new, labels_new = random_deletion(sentence, labels)
                X_new.append(sentence_new)
                Y_new.append(labels_new)
    return X_new, Y_new

def back_translation(X,Y):
    #Back translate the text
    X_new, Y_new = translate_text_to_eng_batch(X,Y)
    X_new, Y_new = translate_text_to_swe_batch(X_new,Y_new)

    #Write the new data to a csv file
    write_csv_file("back_translation", X_new, Y_new, "augmented")
        
    return X_new, Y_new

def random_deletion(sentence, labels):
    dist = random.binomial(n=1, p=binomial_p, size=len(sentence))
    new_sentence = []
    new_labels = []
    for word, label, prob in zip(sentence, labels, dist):
        if prob == 1 and label == 'O':
            continue
        else:
            new_sentence.append(word)
            new_labels.append(label)
    return(new_sentence, new_labels)

def synonym_replacement(sentence, labels):
    generator = LabelGenerator()
    dist = random.binomial(n=1, p=binomial_p, size=len(sentence))
    new_sentence = []
    new_labels = []
    for word, label, prob in zip(sentence, labels, dist):
        if prob == 1 and label == 'O':
            synonym = generator.generate_synonym(word)
            syn_len = len(synonym.split())
            for i in range(syn_len):
                new_sentence.append(synonym.split()[i])
                new_labels.append('O')
        else:
            new_sentence.append(word)
            new_labels.append(label)
    return(new_sentence, new_labels)
            

data = [
    ['ansvarig', 'vid', 'inskrivning', 'är', 'ssk', 'Kalle', 'Ohlsson'],
    ['besök', 'läkare', '19/5', 'på', 'morgonen']
], [
    ['O', 'O', 'O', 'O', 'O', 'B-First_Name', 'B-Last_Name'],
    ['O', 'O', 'B-Date_Part', 'O', 'O']
]

print("len:", len(data[0]))
for col in data:
    for row in col:
        print(row)
new_data = data_augmentation(data[0], data[1], "random_deletion")
print("len:", len(new_data[0]))
for col in new_data:
    for row in col:
        print(row)