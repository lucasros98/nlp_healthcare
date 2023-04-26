import os
import sys
import string
from dotenv import load_dotenv,find_dotenv
from transformers import pipeline
from multiprocessing import Pool
from tqdm import tqdm

load_dotenv(find_dotenv())

#Get the path for the data
PATH = os.getenv('DATA_PATH')

from file_handler import write_csv_file

#Write to train, test and validation data folder
from data import get_training_data

from augmentation import DataAugmentation

#Parameters for data augmentation
class AugmentationParameters():
    data_size_range = [10]
    p_range = [0.1,0.3,0.5]
    num_new_docs_range = [1,2,3]
    methods = ['bert_masking'] ##'synonym_replacement', 'shuffle_within_segments', 'label_wise_token_replacement', 'mention_replacement'] # ['back_translation']

aug_params = AugmentationParameters()
data_aug = DataAugmentation()


def print_augmentation_results(args):
    print("\nData augmentation results:")
    for arg in args:
        print(f'{arg}: {args[arg]}')
    print("")
    

def augment_data(args):
    X_train, Y_train, data_size, p, num_new_docs, aug_method = args
    X_new, Y_new = [], []
    for tokens, labels in tqdm(zip(X_train, Y_train)):
        for i in range(num_new_docs):
            if aug_method == "synonym_replacement":
                X_tmp, Y_tmp = data_aug.synonym_replacement(tokens, labels, p)
            elif aug_method == "random_deletion":
                X_tmp, Y_tmp = data_aug.random_deletion(tokens, labels, p)
            elif aug_method == "mention_replacement":
                X_tmp, Y_tmp = data_aug.mention_replacement(tokens, labels, p)
            elif aug_method == "shuffle_within_segments":
                X_tmp, Y_tmp = data_aug.shuffle_within_segments(tokens, labels, p)
            elif aug_method == "label_wise_token_replacement":
                label_dict = data_aug.create_label_dict(X_train, Y_train)
                X_tmp, Y_tmp = data_aug.label_wise_token_replacement(tokens, labels, p, label_dict)  


            X_new.append(X_tmp)
            Y_new.append(Y_tmp)
    #Write the new data to a csv file
    write_csv_file(aug_method + "_s" + str(num_new_docs) + "_p" + str(p) + "_d" + str(data_size), X_new, Y_new, "augmented")
    print_augmentation_results({"aug_method": aug_method, "num_new_docs": num_new_docs, "p": p, "data_size": data_size, "new_data_length": len(X_new)})


for data_size in aug_params.data_size_range:
    for aug_method in aug_params.methods:
        X_train,Y_train,_,_,_,_ = get_training_data(precentage=data_size)
        print("Data length: ", str(len(X_train)))

        # Run back-translation for each data size and number of new documents   
        if "back_translation" == aug_params.methods:
            print("Starting back-translation...")
            X_new, Y_new = [], []
            for num_new_docs in aug_params.num_new_docs_range:
                if num_new_docs == 1:
                    X_new, Y_new = X_train, Y_train
            
                X_tmp, Y_tmp = data_aug.back_translation(X_new, Y_new)
                X_new.extend(X_tmp)
                Y_new.extend(Y_tmp)

                write_csv_file("back_translation" + "_s" + str(num_new_docs) + "_d" + str(data_size), X_new, Y_new, "augmented")
                print_augmentation_results({"aug_method": "back_translation", "num_new_docs": num_new_docs, "data_size": data_size})
       
        elif "bert_masking" == aug_method:
            bert_mask = pipeline('fill-mask', model='KB/bert-base-swedish-cased')

            for num_new_docs in aug_params.num_new_docs_range:
                for p in aug_params.p_range:
                    X_new, Y_new = [], []

                    for tokens, labels in tqdm(zip(X_train, Y_train)):
                        for i in range(num_new_docs):
                            X_tmp, Y_tmp = data_aug.bert_masking_single(tokens, labels, p, bert_mask=bert_mask)
                            X_new.append(X_tmp)
                            Y_new.append(Y_tmp)
                    write_csv_file(aug_method + "_s" + str(num_new_docs) + "_p" + str(p) + "_d" + str(data_size), X_new, Y_new, "augmented")
        
        else:
            # Run each specified aug method (exepct back_translation) for each data size, p and number of new documents
            # Use a Pool to parallelize the outermost loop
            with Pool() as pool:
                pool.map(augment_data, [(X_train, Y_train, data_size, p, num_new_docs, aug_method) for p in aug_params.p_range for num_new_docs in aug_params.num_new_docs_range])


print("Data augmentation done jihooo!")

# X_train, Y_train = [
#         ['besök', 'av', 'moder', 'Lucas', '80', 'år', '19/5', '-18', 'på', 'morgonen'],
#         ['ansvarig', 'vid', 'inskrivning', 'är', 'ssk', 'Kim', 'Dahlström', 'Johansson', 'Kallesson'],
#     ], [
#         ['O', 'O', 'O', 'B-First_Name', 'B-Age', 'I-Age', 'B-Full_Date', 'I-Full_Date', 'O', 'O'],
#         ['O', 'O', 'O', 'O', 'O', 'B-First_Name', 'B-Last_Name', 'I-Last_Name', 'I-Last_Name'],
#     ]