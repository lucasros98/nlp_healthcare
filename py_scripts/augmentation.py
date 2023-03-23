# Functions for data augmentation
import numpy as np
from numpy import random
from translation import translate_text_to_eng_batch, translate_text_to_swe_batch
from file_handler import read_csv_file, write_csv_file
from generation import LabelGenerator
from tqdm import tqdm
from data import get_training_data
from preprocessing import remove_duplicates
import os
import sys
from dotenv import load_dotenv,find_dotenv
from multiprocessing import Pool

sys.path.append(os.path.dirname(find_dotenv()))
load_dotenv(find_dotenv())

#Get the path for the data
PATH = os.getenv('DATA_PATH')

class DataAugmentation():
    def __init__(self, X, Y, aug_type, binomial_p, num_new_docs, data_size):
        self.generator = LabelGenerator()
        self.binomial_p = binomial_p
        self.num_new_docs = num_new_docs
        self.X = X
        self.Y = Y
        self.aug_type = aug_type
        self.data_size = data_size

    def augment_data(self):
        X_new, Y_new = [], []
        
        for tokens, labels in tqdm(zip(self.X, self.Y)):
            for i in range(self.num_new_docs):
                if self.aug_type == "synonym_replacement":
                    tokens_new, labels_new = self.synonym_replacement(tokens, labels)
                elif self.aug_type == "random_deletion":
                    tokens_new, labels_new = self.random_deletion(tokens, labels)
                elif self.aug_type == "mention_replacement":
                    tokens_new, labels_new = self.mention_replacement(tokens, labels)
                elif self.aug_type == "shuffle_within_segments":
                    tokens_new, labels_new = self.shuffle_within_segments(tokens, labels)
                elif self.aug_type == "label_wise_token_replacement":
                    label_dict = self.create_label_dict(self.X, self.Y)
                    tokens_new, labels_new = self.label_wise_token_replacement(tokens, labels, label_dict)  
                X_new.append(tokens_new)
                Y_new.append(labels_new)
        #Write the new data to a csv file
        write_csv_file(self.aug_type + "_s" + str(self.num_new_docs) + "_p" + str(self.binomial_p) + "_d" + str(self.data_size), X_new, Y_new, "augmented")
        
        print(f'Data augmentation {self.aug_type}, num new docs {self.num_new_docs}, p {self.binomial_p}, data size {self.data_size}: DONE')
        print(f'New documents created: {len(X_new)}')
        print('')


    def back_translation(self):
        #Back translate the text
        X_new, Y_new = translate_text_to_eng_batch(self.X, self.Y)
        X_new, Y_new = translate_text_to_swe_batch(X_new,Y_new)

        #Write the new data to a csv file
        write_csv_file("back_translation" + "_d" + str(self.data_size), X_new, Y_new, "augmented")
        print(f'Data augmentation back_translation, data size {self.data_size}: DONE')
        print(f'New documents created: {len(X_new)}')
        print('')
        return X_new, Y_new


    def get_gender(self, prev_token, first_name):
        if prev_token in ['mamma', 'mamman', 'mor', 'modern', 'syster', 'systern', 'mormor', 'farmor', 'dotter', 'dottern', 'fru', 'frun', 'hustru', 'hustrun', 'brud', 'bruden', 'faster', 'fastern', 'moster', 'mostern']:
            return 'woman'
        elif prev_token in ['pappa', 'pappan', 'far', 'fadern', 'bror', 'brodern', 'morfar', 'farfar', 'son', 'sonen', 'herr', 'herren', 'man', 'make', 'maken', 'brudgum', 'farbror', 'farbrorn' 'morbror', 'morbrorn']:
            return 'man'
        else:
            return self.generator.get_gender_of_first_name(first_name)


    def mention_replacement(self, X, Y):
        merged_tokens = []
        merged_labels = []
        # merge mentions that are split into multiple tokens (e.g. "B-First_Name, I-First_Name" -> "B-First_Name")
        for i, (token, label) in enumerate(zip(X, Y)):
            if label != 'O':
                if i > 0 and Y[i-1][2:] == label[2:]:
                    merged_tokens[-1] = merged_tokens[-1] + ' ' + token
                else:
                    merged_tokens.append(token)
                    merged_labels.append(label)
            else:
                merged_tokens.append(token)
                merged_labels.append(label)

        X_new = []
        Y_new = []
        mentions = list(filter(lambda x: x != 'O', merged_labels)) # get all labels that are not 'O'
        dist = random.binomial(n=1, p=self.binomial_p, size=len(mentions)) # distribution of mentions to replace
        mention_index = 0
        mentions_replaced = 0

        for i, (token, label) in enumerate(zip(merged_tokens, merged_labels)):
            gender = None
            if label != 'O':
                # special case 1: get gender of first name
                if label == 'B-First_Name' and i > 0:
                    gender = self.get_gender(X[i-1], token)
                if dist[mention_index] == 1: 
                    new_mention = str(self.generator.generate_random_entity(label[2:], params={"gender": gender}))
                    new_mention_len = len(new_mention.split()) # get number of tokens in generated mention
                    for j in range(new_mention_len):
                        X_new.append(new_mention.split()[j])
                        if j > 0:
                            Y_new.append('I-' + label[2:])
                        else:
                            Y_new.append(label)
                    mention_index += 1
                    mentions_replaced += 1
                else:
                    X_new.append(token)
                    Y_new.append(label)
            else:
                X_new.append(token)
                Y_new.append(label)

        return(X_new, Y_new)


    def random_deletion(self, X, Y):
        dist = random.binomial(n=1, p=self.binomial_p, size=len(X))
        X_new = []
        Y_new = []
        # delete tokens of label 'O' by probability p
        for token, label, prob in zip(X, Y, dist):
            if prob == 1 and label == 'O':
                continue
            else:
                X_new.append(token)
                Y_new.append(label)
        return(X_new, Y_new)


    def synonym_replacement(self, X, Y):
        dist = random.binomial(n=1, p=self.binomial_p, size=len(X))
        X_new = []
        Y_new = []
        # replace tokens of label 'O' with synonyms by probability p
        for token, label, prob in zip(X, Y, dist):
            if prob == 1 and label == 'O':
                synonym = self.generator.generate_synonym(token)
                syn_len = len(synonym.split()) # get number of tokens in generated synonym
                for i in range(syn_len):
                    X_new.append(synonym.split()[i])
                    Y_new.append('O')
            else:
                X_new.append(token)
                Y_new.append(label)
        return(X_new, Y_new)

    # Old version (shuffles labels as well)
    def shuffle_within_segments_old(self, X, Y):
        X_new = []
        Y_new = []
        segments = []
        prev_label = ''
        for token, label in zip(X, Y):
            _label = label if label == 'O' else label[2:]
            if _label != prev_label:
                segments.append([(token, label)])
                prev_label = _label
            else:
                segments[-1].append((token, label))

        # shuffle tokens and labels within segments with probability p
        dist = random.binomial(n=1, p=self.binomial_p, size=len(segments))
        for i, prob in enumerate(dist):
            if prob == 1:
                random.shuffle(segments[i])
        
        # merge segments into new sentence
        for segment in segments:
            for token, label in segment:
                X_new.append(token)
                Y_new.append(label)

        return(X_new, Y_new)

    def shuffle_within_segments(self, X, Y):
        X_new = []
        Y_new = []
        segments = []
        prev_label = ''
        for token, label in zip(X, Y):
            _label = label if label == 'O' else label[2:]
            if _label != prev_label:
                segments.append([token])
                prev_label = _label
            else:
                segments[-1].append(token)

        # shuffle tokens within segments with probability p
        dist = random.binomial(n=1, p=self.binomial_p, size=len(segments))
        for i, prob in enumerate(dist):
            if prob == 1:
                random.shuffle(segments[i])
        
        # merge segments into new sentence 
        for segment in segments:
            for token in segment:
                X_new.append(token)
        
        # add labels to new sentence (not shuffled)
        Y_new = Y
        return(X_new, Y_new)

    # create dict with labels as keys and list of tokens as values
    def create_label_dict(self, X, Y):
        label_dict = {}
        for tokens, labels in zip(X, Y):
            for token, label in zip(tokens, labels):
                _label = label if label == 'O' else label[2:]
                if _label in label_dict:
                    label_dict[_label].append(token)
                else:
                    label_dict[_label] = [token]
        return label_dict

    def label_wise_token_replacement(self, X, Y, label_dict):
        X_new = []
        Y_new = []
        dist = random.binomial(n=1, p=self.binomial_p, size=len(X))

        # replace tokens with random tokens of the same label
        for token, label, prob in zip(X, Y, dist):
            _label = label if label == 'O' else label[2:]
            if prob == 1:
                X_new.append(random.choice(label_dict[_label]))
                Y_new.append(label)
            else:
                X_new.append(token)
                Y_new.append(label)
        return(X_new, Y_new)

data = [
    ['besök', 'mamma', 'Lucas', '80', 'år', '19/5', '-18', 'på', 'morgonen'],
    ['ansvarig', 'vid', 'inskrivning', 'är', 'ssk', 'Kim', 'Dahlström', 'Johansson', 'Kallesson'],
], [
    ['O', 'O', 'B-First_Name', 'B-Age', 'I-Age', 'B-Full_Date', 'I-Full_Date', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'O', 'B-First_Name', 'B-Last_Name', 'I-Last_Name', 'I-Last_Name'],
]

#Data augmentation
data_size_range = [10, 25, 50, 75, 100]
p_range = [0.1, 0.3, 0.5, 0.7]
num_new_docs_range = [1, 3, 6, 10]
aug_methods = ['random_deletion', 'synonym_replacement', 'shuffle_within_segments', 'label_wise_token_replacement', 'mention_replacement']

def run_data_augmentation(args):
    data_size, p, num_new_docs, aug_method = args
    X_train,Y_train,_,_,_,_ = get_training_data(precentage=data_size)
    data_aug = DataAugmentation(X_train, Y_train, aug_method, binomial_p=p, num_new_docs=num_new_docs, data_size=data_size)
    data_aug.augment_data()

print("Starting data augmentation...")
# Use a Pool to parallelize the outermost loop
with Pool() as p:
    p.map(run_data_augmentation, [(data_size, p, num_new_docs, aug_method) for data_size in data_size_range for p in p_range for num_new_docs in num_new_docs_range for aug_method in aug_methods])

print("Starting back-translation...")
# Run back-translation for each data size
for data_size in data_size_range:
    X_train,Y_train,_,_,_,_ = get_training_data(precentage=data_size)
    data_aug = DataAugmentation(X_train, Y_train, aug_type="back-translation", binomial_p=1, num_new_docs=1, data_size=data_size)
    data_aug.back_translation()

print("Data augmentation done jihooo!")
