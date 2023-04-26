# Functions for data augmentation
import numpy as np
from numpy import random
from translation import translate_text_to_eng_batch, translate_text_to_swe_batch
from generation import LabelGenerator
import copy
import os
import sys
from dotenv import load_dotenv,find_dotenv
from data import get_all_entities
from file_handler import read_csv_file

sys.path.append(os.path.dirname(find_dotenv()))
load_dotenv(find_dotenv())

#Get the path for the data
PATH = os.getenv('DATA_PATH')

#Parameters for translation models
class TranslationParametersENG():
    num_beams=4
    early_stopping=True
    max_length=512
    use_decoded=True
    type='s4s'

#Parameters for translation models
class TranslationParametersSWE():
    num_beams=4
    early_stopping=True
    max_length=512
    use_decoded=True

class DataAugmentation():
    def __init__(self):
        self.generator = LabelGenerator()

    # Translate the whole dataset to english and back to swedish
    def back_translation(self, X, Y, num_sentences=1, bt_type='s4s'):
        # Parameters for translation models
        params_eng = TranslationParametersENG()
        params_swe = TranslationParametersSWE()

        _X = copy.deepcopy(X)
        _Y = copy.deepcopy(Y)

        # Translate to english
        params_eng.num_sentences = 1
        X_new, Y_new = translate_text_to_eng_batch(_X, _Y, params=params_eng)

        # Translate back to swedish
        params_swe.num_sentences = num_sentences # number of sentences to generate for each sentence
        params_swe.type = bt_type # 's4s' or 'w4w'
        X_new, Y_new = translate_text_to_swe_batch(X_new,Y_new,params=params_swe)
        return X_new, Y_new


    # Helper function for mention_replacement
    def get_gender(self, prev_token, first_name):
        if prev_token in ['mamma', 'mamman', 'mor', 'modern', 'syster', 'systern', 'mormor', 'farmor', 'dotter', 'dottern', 'fru', 'frun', 'hustru', 'hustrun', 'brud', 'bruden', 'faster', 'fastern', 'moster', 'mostern']:
            return 'woman'
        elif prev_token in ['pappa', 'pappan', 'far', 'fadern', 'bror', 'brodern', 'morfar', 'farfar', 'son', 'sonen', 'herr', 'herren', 'man', 'make', 'maken', 'brudgum', 'farbror', 'farbrorn' 'morbror', 'morbrorn']:
            return 'man'
        else:
            return self.generator.get_gender_of_first_name(first_name)


    # Replace entities not labeled as 'O' (mentions) with new generated entities
    def mention_replacement(self, X, Y, p):
        # Get the unique test data
        X_unique, Y_unique = read_csv_file('test_sv_unique.csv', subfolder="test")
        #Create blacklist for entities from unique test data
        black_list = get_all_entities(X_unique, Y_unique)
        #Remove the common entities present in the blacklist (so that we don't generate them)
        for key in black_list:
            black_list[key] = list(set(black_list[key]))
            self.generator.remove_common_entities(black_list[key],key)
    

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
        dist = random.binomial(n=1, p=p, size=len(mentions)) # distribution of mentions to replace
        mention_index = 0

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
                else:
                    token_len = len(token.split()) # get number of subtokens in mention
                    for j in range(token_len):
                        X_new.append(token.split()[j])
                        if j > 0:
                            Y_new.append('I-' + label[2:])
                        else:
                            Y_new.append(label)
            else:
                X_new.append(token)
                Y_new.append(label)
        return X_new, Y_new


    # Delete tokens of label 'O' by probability p
    def random_deletion(self, X, Y, p):
        dist = random.binomial(n=1, p=p, size=len(X))
        X_new = []
        Y_new = []
        
        for token, label, prob in zip(X, Y, dist):
            if prob == 1 and label == 'O':
                continue
            else:
                X_new.append(token)
                Y_new.append(label)
        return X_new, Y_new

    # Replace tokens of label 'O' with synonyms by probability p
    def synonym_replacement(self, X, Y, p):
        dist = random.binomial(n=1, p=p, size=len(X))
        X_new = []
        Y_new = []
        
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
        return X_new, Y_new


    # Old version (shuffles labels as well)
    def shuffle_within_segments_old(self, X, Y, p):
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
        dist = random.binomial(n=1, p=p, size=len(segments))
        for i, prob in enumerate(dist):
            if prob == 1:
                random.shuffle(segments[i])
        
        # merge segments into new sentence
        for segment in segments:
            for token, label in segment:
                X_new.append(token)
                Y_new.append(label)
        return X_new, Y_new

    
    # Divide sentence into segments based on labels, then shuffle tokens within segments
    def shuffle_within_segments(self, X, Y, p):
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
        dist = random.binomial(n=1, p=p, size=len(segments))
        for i, prob in enumerate(dist):
            if prob == 1:
                random.shuffle(segments[i])
        
        # merge segments into new sentence 
        for segment in segments:
            for token in segment:
                X_new.append(token)
        
        # add labels to new sentence (not shuffled)
        Y_new = Y
        return X_new, Y_new


    # Create dict with labels as keys and list of tokens as values
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


    # Replace tokens with random tokens of the label 'O'
    def label_wise_token_replacement(self, X, Y, p, label_dict):
        X_new = []
        Y_new = []
        dist = random.binomial(n=1, p=p, size=len(X))

        # replace tokens with random tokens of the same label
        for token, label, prob in zip(X, Y, dist):
            _label = label if label == 'O' else label[2:]
            if prob == 1:
                X_new.append(random.choice(label_dict[_label]))
                Y_new.append(label)
            else:
                X_new.append(token)
                Y_new.append(label)
        return X_new, Y_new 

    
    # Replace mentions with random mentions of the same label
    def local_mention_replacement(self, X, Y, p, label_dict):
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
        dist = random.binomial(n=1, p=p, size=len(mentions)) # distribution of mentions to replace
        mention_index = 0

        for i, (token, label) in enumerate(zip(merged_tokens, merged_labels)):
            _label = label if label == 'O' else label[2:]
            if _label != 'O':
                if dist[mention_index] == 1: 
                    new_mention = random.choice(label_dict[_label])
                    new_mention_len = len(new_mention.split()) # get number of tokens in generated mention
                    for j in range(new_mention_len):
                        X_new.append(new_mention.split()[j])
                        if j > 0:
                            Y_new.append('I-' + label[2:])
                        else:
                            Y_new.append(label)
                    mention_index += 1
                else:
                    token_len = len(token.split()) # get number of subtokens in mention
                    for j in range(token_len):
                        X_new.append(token.split()[j])
                        if j > 0:
                            Y_new.append('I-' + label[2:])
                        else:
                            Y_new.append(label)
            else:
                X_new.append(token)
                Y_new.append(label)
        return X_new, Y_new
