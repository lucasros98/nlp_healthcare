import os
import sys
import string
from dotenv import load_dotenv,find_dotenv

load_dotenv(find_dotenv())

#Get the path for the data
PATH = os.getenv('DATA_PATH')

from file_handler import write_csv_file

#Write to train, test and validation data folder
from data import get_training_data

#For translation 
from translation import translate_text_to_eng_batch

#Parameters for translation models
class TranslationParameters():
    num_beams=4
    early_stopping=True
    max_length=512
    use_decoded=True
    num_sentences=1

translations_params = TranslationParameters()

for i in [10,25,50,75,100]:
    X_train, Y_train, X_val, Y_val, X_test, Y_test = get_training_data(precentage=i)

    X_train_en, Y_train_en = translate_text_to_eng_batch(X_train,Y_train,params=translations_params)

    #Write to files
    name="train_en_"+str(i)
    write_csv_file(filename=name,X=X_train_en,Y=Y_train_en,subfolder="train")

X_val_en, Y_val_en = translate_text_to_eng_batch(X_val,Y_val,params=translations_params)
X_test_en, Y_test_en = translate_text_to_eng_batch(X_test,Y_test,params=translations_params)

write_csv_file(filename="val_en",X=X_val_en,Y=Y_val_en,subfolder="val")
write_csv_file(filename="test_en",X=X_test_en,Y=Y_test_en,subfolder="test")

