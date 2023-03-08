# Functions for data augmentation
import numpy as np
from numpy import random
#from translation import translate_text_to_eng_batch, translate_text_to_swe_batch
from file_handler import read_csv_file, write_csv_file
from generation import LabelGenerator
from tqdm import tqdm
from data import get_training_data
import os
import sys
from dotenv import load_dotenv,find_dotenv

sys.path.append(os.path.dirname(find_dotenv()))
load_dotenv(find_dotenv())

#Get the path for the data
PATH = os.getenv('DATA_PATH')

binomial_p = 0.3
num_new_docs = 1

def data_augmentation(aug_type):
    X_train,Y_train,X_val,Y_val,X_test,Y_test = get_training_data(precentage=25,uncased=False)
    X_new = X_train.copy()
    Y_new = Y_train.copy()

    print('Start: X_train length: ', len(X_train))
    print('Start: Y_train length: ', len(Y_train))

    for sentence, labels in tqdm(zip(X_train,Y_train), desc=f'aug_type={aug_type} with p={binomial_p}, num_new_docs={num_new_docs}'):
        for i in range(num_new_docs):
            if aug_type == "synonym_replacement":
                sentence_new, labels_new = synonym_replacement(sentence, labels)
                X_new.append(sentence_new)
                Y_new.append(labels_new)
            elif aug_type == "random_deletion":
                sentence_new, labels_new = random_deletion(sentence, labels)
                X_new.append(sentence_new)
                Y_new.append(labels_new)
            elif aug_type == "mention_replacement":
                sentence_new, labels_new = mention_replacement(sentence, labels)
                X_new.append(sentence_new)
                Y_new.append(labels_new)
    

    print('End: X_train length before duplicates removed: ', len(X_new))
    print('End: Y_train length: before duplicates removed', len(Y_new))

    # remove duplicates from X_new and Y_new
    X_new = list(dict.fromkeys(X_new))
    Y_new = list(dict.fromkeys(Y_new))

    print('End: X_train length after duplicates removed: ', len(X_new))
    print('End: Y_train length: after duplicates removed', len(Y_new))

    #Write the new data to a csv file
    write_csv_file(aug_type, X_new, Y_new, "augmented")

    return X_new, Y_new

def back_translation(X,Y):
    #Back translate the text
    X_new, Y_new = translate_text_to_eng_batch(X,Y)
    X_new, Y_new = translate_text_to_swe_batch(X_new,Y_new)

    #Write the new data to a csv file
    write_csv_file("back_translation", X_new, Y_new, "augmented")
        
    return X_new, Y_new

def get_gender_of_prev_word(word):
    if word in ['mamma', 'mamman', 'mor', 'modern', 'syster', 'systern', 'mormor', 'farmor', 'dotter', 'dottern', 'fru', 'frun', 'hustru', 'hustrun', 'brud', 'bruden', 'faster', 'fastern', 'moster', 'mostern']:
        return 'woman'
    elif word in ['pappa', 'pappan', 'far', 'fadern', 'bror', 'brodern', 'morfar', 'farfar', 'son', 'sonen', 'herr', 'herren', 'man', 'make', 'maken', 'brudgum', 'farbror', 'farbrorn' 'morbror', 'morbrorn']:
        return 'man'
    else:
        return None

def mention_replacement(sentence, labels):
    generator = LabelGenerator()
    new_sentence = []
    new_labels = []

    mentions = list(filter(lambda x: x != 'O', labels))
    num_mentions = len(mentions)
    dist = random.binomial(n=1, p=binomial_p, size=num_mentions) # distribution of mentions to replace
    mention_index = 0

    for i, (word, label) in enumerate(zip(sentence, labels)):
        gender = None
        if label != 'O':
            if label[2:] == 'First_Name' and i > 0:
                gender = get_gender_of_prev_word(sentence[i-1])
                if gender == None:
                    gender = generator.get_gender_of_first_name(word)
            if dist[mention_index] == 1: 
                new_mention = generator.generate_random_entity(label[2:], params={"gender": gender})
                new_mention_len = len(new_mention.split()) # get number of words in new mention
                for j in range(new_mention_len):
                    new_sentence.append(new_mention.split()[j])
                    if j > 0:
                        new_labels.append('I-' + label[2:])
                    else:
                        new_labels.append(label)
                mention_index += 1
            else:
                new_sentence.append(word)
                new_labels.append(label)
        else:
            new_sentence.append(word)
            new_labels.append(label)
    return(new_sentence, new_labels)

        


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
            syn_len = len(synonym.split()) # get number of words in synonym
            for i in range(syn_len):
                new_sentence.append(synonym.split()[i])
                new_labels.append('O')
        else:
            new_sentence.append(word)
            new_labels.append(label)
    return(new_sentence, new_labels)
            


data_augmentation("synonym_replacement")
