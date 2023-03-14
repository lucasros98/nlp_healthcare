# Functions for data augmentation
import numpy as np
from numpy import random
#from translation import translate_text_to_eng_batch, translate_text_to_swe_batch
from file_handler import read_csv_file, write_csv_file
from generation import LabelGenerator
from tqdm import tqdm
from data import get_training_data
from preprocessing import remove_duplicates
import os
import sys
from dotenv import load_dotenv,find_dotenv

sys.path.append(os.path.dirname(find_dotenv()))
load_dotenv(find_dotenv())

#Get the path for the data
PATH = os.getenv('DATA_PATH')

class DataAugmentation():
    def __init__(self, X, Y, aug_type):
        self.generator = LabelGenerator()
        self.binomial_p = 0.5
        self.num_new_docs = 1
        self.X = X
        self.Y = Y
        self.aug_type = aug_type

    def augment_data(self):
        X_new = []
        Y_new = []

        print(f'aug_type={self.aug_type} with p={self.binomial_p}, num_new_docs={self.num_new_docs}')

        for sentence, labels in tqdm(zip(self.X, self.Y)):
            for i in range(self.num_new_docs):
                if self.aug_type == "synonym_replacement":
                    sentence_new, labels_new = self.synonym_replacement(sentence, labels)
                elif self.aug_type == "random_deletion":
                    sentence_new, labels_new = self.random_deletion(sentence, labels)
                elif self.aug_type == "mention_replacement":
                    sentence_new, labels_new = self.mention_replacement(sentence, labels)
                elif self.aug_type == "shuffle_within_segments":
                    sentence_new, labels_new = self.shuffle_within_segments(sentence, labels)
                X_new.append(sentence_new)
                Y_new.append(labels_new)    

        #Write the new data to a csv file
        write_csv_file(self.aug_type, X_new, Y_new, "augmented")
        
        print(f'Data augmentation {self.aug_type}: done')
        print(f'Number of new documents: {len(X_new)}')
        
        return X_new, Y_new

    def back_translation(self, X, Y):
        #Back translate the text
        X_new, Y_new = translate_text_to_eng_batch(self.X, self.Y)
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
        new_sentence = []
        new_labels = []

        mentions = list(filter(lambda x: x != 'O', labels)) # get all labels that are not 'O'
        num_mentions = len(mentions)
        dist = random.binomial(n=1, p=self.binomial_p, size=num_mentions) # distribution of mentions to replace
        mention_index = 0
        mentions_replaced = 0

        for i, (word, label) in enumerate(zip(sentence, labels)):
            gender = None
            if label != 'O':
                if label[2:] == 'First_Name' and i > 0:
                    gender = get_gender_of_prev_word(sentence[i-1])
                    if gender == None:
                        gender = self.generator.get_gender_of_first_name(word)
                if dist[mention_index] == 1: 
                    new_mention = self.generator.generate_random_entity(label[2:], params={"gender": gender})
                    new_mention_len = len(new_mention.split()) # get number of words in new mention
                    for j in range(new_mention_len):
                        new_sentence.append(new_mention.split()[j])
                        if j > 0:
                            new_labels.append('I-' + label[2:])
                        else:
                            new_labels.append(label)
                    mention_index += 1
                    mentions_replaced += 1
                else:
                    new_sentence.append(word)
                    new_labels.append(label)
            else:
                new_sentence.append(word)
                new_labels.append(label)

        return(new_sentence, new_labels)


    def random_deletion(sentence, labels):
        dist = random.binomial(n=1, p=self.binomial_p, size=len(sentence))
        new_sentence = []
        new_labels = []
        for word, label, prob in zip(sentence, labels, dist):
            if prob == 1 and label == 'O':
                continue
            else:
                new_sentence.append(word)
                new_labels.append(label)
        return(new_sentence, new_labels)


    def synonym_replacement(self, sentence, labels):
        dist = random.binomial(n=1, p=self.binomial_p, size=len(sentence))
        new_sentence = []
        new_labels = []
        for word, label, prob in zip(sentence, labels, dist):
            if prob == 1 and label == 'O':
                synonym = self.generator.generate_synonym(word)
                syn_len = len(synonym.split()) # get number of words in synonym
                for i in range(syn_len):
                    new_sentence.append(synonym.split()[i])
                    new_labels.append('O')
            else:
                new_sentence.append(word)
                new_labels.append(label)
        return(new_sentence, new_labels)


    def shuffle_within_segments(self, sentence, labels):
        new_sentence = []
        new_labels = []
        segments = []
        prev_label = ''
        for word, label in zip(sentence, labels):
            _label = label if label == 'O' else label[2:]
            if _label != prev_label:
                segments.append([(word, label)])
                prev_label = _label
            else:
                segments[-1].append((word, label))

        dist = random.binomial(n=1, p=self.binomial_p, size=len(segments))
        for i, prob in enumerate(dist):
            if prob == 1:
                random.shuffle(segments[i])
        
        for segment in segments:
            for word, label in segment:
                new_sentence.append(word)
                new_labels.append(label)

        return(new_sentence, new_labels)

data = [
    ['ansvarig', 'vid', 'inskrivning', 'är', 'ssk', 'Kalle', 'Ohlsson'],
    ['besök', 'läkare', '19/5', 'på', 'morgonen'],
    ['ansvarig', 'vid', 'inskrivning', 'är', 'ssk', 'Kim', 'Dahlström', 'Johansson', 'Kallesson'],
    ['besök', 'mamma', 'Lucas', 'på', 'morgonen']
], [
    ['O', 'O', 'O', 'O', 'O', 'B-First_Name', 'B-Last_Name'],
    ['O', 'O', 'B-Date_Part', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'O', 'B-First_Name', 'B-Last_Name', 'I-Last_Name', 'I-Last_Name'],
    ['O', 'O', 'B-First_Name', 'O', 'O']
]
    
X_train,Y_train,X_val,Y_val,X_test,Y_test = get_training_data(precentage=25,uncased=False)

#Data augmentation
data_aug = DataAugmentation(X_train, Y_train, "synonym_replacement")
X_train, Y_train = data_aug.augment_data()
