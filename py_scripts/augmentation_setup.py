import os
import sys
import string
from dotenv import load_dotenv,find_dotenv
from transformers import pipeline
from multiprocessing import Pool
from tqdm import tqdm
from data import get_all_entities

load_dotenv(find_dotenv())

#Get the path for the data
PATH = os.getenv('DATA_PATH')

from file_handler import write_csv_file

from data import get_training_data, get_augmented_data, build_file_name

from augmentation import DataAugmentation

#Parameters for data augmentation
class AugmentationParameters():
    data_size_range = [10]
    p_range = [0.3]
    num_new_docs_range = [1]
    methods = ['unique_mention_replacement'] # ['random_deletion', 'synonym_replacement', 'shuffle_within_segments', 'label_wise_token_replacement', 'unique_mention_replacement'] # ['back_translation']
    merge_methods = True

class AugmentationParametersMergeBase():
    data_size = 10
    p = None
    num_new_docs = 2
    aug_method = 'back_translation_s4s' # ['random_deletion', 'synonym_replacement', 'shuffle_within_segments', 'label_wise_token_replacement', 'unique_mention_replacement'] # ['back_translation']

aug_params = AugmentationParameters()
params_merge_base = AugmentationParametersMergeBase()
data_aug = DataAugmentation()


def print_augmentation_results(args):
    print("\nData augmentation results:")
    for arg in args:
        print(f'{arg}: {args[arg]}')
    print("")


def get_data(data_size):
    if aug_params.merge_methods: 
        X_train, Y_train = get_augmented_data({'data_size': data_size, 'p': params_merge_base.p, 'num_new_docs': params_merge_base.num_new_docs, 'aug_method': params_merge_base.aug_method})
    else:
        X_train,Y_train,_,_,_,_ = get_training_data(precentage=data_size)
      
    print("Data length: ", str(len(X_train)))
    return X_train,Y_train

def write_data(args, X_new, Y_new):
    #Write the new data to a csv file
    if not aug_params.merge_methods:
        file_name = build_file_name(**args)
        subfolder = "augmented"
    else:
        file_name = build_file_name(**{'aug_method': [params_merge_base.aug_method, args['aug_method']], 'num_new_docs': [params_merge_base.num_new_docs, args['num_new_docs']], 'data_size': args['data_size'], 'p': [params_merge_base.p, args['p']]})
        subfolder = "augmented_merged"

    write_csv_file(file_name, X_new, Y_new, subfolder)
    print_augmentation_results(args)


def augment_data(args):
    aug_method = args['aug_method']
    num_new_docs = args['num_new_docs']
    p = args['p']
    data_size = args['data_size']

    X_train, Y_train = get_data(data_size)
    X_new, Y_new = [], []
    for tokens, labels in tqdm(zip(X_train, Y_train)):
        for _ in range(num_new_docs):
            if aug_method == "synonym_replacement":
                X_tmp, Y_tmp = data_aug.synonym_replacement(tokens, labels, p)
            elif aug_method == "random_deletion":
                X_tmp, Y_tmp = data_aug.random_deletion(tokens, labels, p)
            elif aug_method == "unique_mention_replacement":
                X_tmp, Y_tmp = data_aug.mention_replacement(tokens, labels, p)
            elif aug_method == "local_mention_replacement":
                label_dict = get_all_entities(X_train, Y_train)
                X_tmp, Y_tmp = data_aug.local_mention_replacement(tokens, labels, p, label_dict)
            elif aug_method == "shuffle_within_segments":
                X_tmp, Y_tmp = data_aug.shuffle_within_segments(tokens, labels, p)
            elif aug_method == "label_wise_token_replacement":
                label_dict = data_aug.create_label_dict(X_train, Y_train)
                X_tmp, Y_tmp = data_aug.label_wise_token_replacement(tokens, labels, p, label_dict)  


            X_new.append(X_tmp)
            Y_new.append(Y_tmp)

    write_data(args, X_new, Y_new)


for data_size in aug_params.data_size_range:
    for aug_method in aug_params.methods:
        # Run back-translation for each data size and number of new documents   
        if(aug_method[:-4] == "back_translation"):
            X_train, Y_train = get_data(data_size)

            for num_new_docs in aug_params.num_new_docs_range:
                X_new, Y_new = data_aug.back_translation(X_train, Y_train, num_sentences=num_new_docs, bt_type=aug_method[-3:])

                params_dict = {
                    'aug_method': aug_method,
                    'num_new_docs': num_new_docs,
                    'data_size': data_size
                }

                write_data(params_dict, X_new, Y_new)

        elif "bert_masking" == aug_method:
            X_train, Y_train = get_data(data_size)
            bert_mask = pipeline('fill-mask', model='KB/bert-base-swedish-cased')

            for num_new_docs in aug_params.num_new_docs_range:
                for p in aug_params.p_range:
                    X_new, Y_new = [], []

                    for tokens, labels in tqdm(zip(X_train, Y_train)):
                        for i in range(num_new_docs):
                            X_tmp, Y_tmp = data_aug.bert_masking_single(tokens, labels, p, bert_mask=bert_mask)
                            X_new.append(X_tmp)
                            Y_new.append(Y_tmp)

                    params_dict = {
                        'aug_method': aug_method,
                        'num_new_docs': num_new_docs,
                        'data_size': data_size,
                        'p': p
                    }
                    write_data(params_dict, X_new, Y_new)
        
        else:
            # Run each specified aug method (exepct back_translation) for each data size, p and number of new documents
            # Use a Pool to parallelize the outermost loop
            with Pool() as pool:
                pool.map(augment_data, [{'data_size': data_size, 'p': p, 'num_new_docs': num_new_docs, 'aug_method': aug_method} for p in aug_params.p_range for num_new_docs in aug_params.num_new_docs_range])


print("Data augmentation done jihooo!")

# X_train, Y_train = [
#         ['besök', 'av', 'moder', 'Lucas', '80', 'år', '19/5', '-18', 'på', 'morgonen'],
#         ['100-årig', 'ansvarig', 'vid', 'inskrivning', '20/01', 'är', 'ssk', 'Kim', 'Johan', 'Dahlström', 'Johansson', 'Kallesson'],
#     ], [
#         ['O', 'O', 'O', 'B-First_Name', 'B-Age', 'I-Age', 'B-Full_Date', 'I-Full_Date', 'O', 'O'],
#         ['B-Age', 'O', 'O', 'O', 'O', 'B-Date_Part', 'O', 'B-First_Name', 'I-First_Name', 'B-Last_Name', 'I-Last_Name', 'I-Last_Name'],
#     ]