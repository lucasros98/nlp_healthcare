import os
import sys
import string
from dotenv import load_dotenv,find_dotenv

load_dotenv(find_dotenv())

#Get the path for the data
PATH = os.getenv('DATA_PATH')

from file_handler import write_csv_file
from preprocessing import preprocessing

#Write to train, test and validation data folder
from data import create_data_dirs, split_data, split_randomly, generate_unique_test_data

#For translation 
from translation import translate_text_to_eng_batch

#puncation without - and >
punctuation = string.punctuation.replace('-','')
punctuation = punctuation.replace('>','')

X, Y = preprocessing(IOB=True,punctuation=punctuation,cased=True)

#Create the data directories
create_data_dirs()

#Write to processed data
write_csv_file(filename="cleaned",X=X,Y=Y,subfolder="processed")

#Create all the data
#Split the data
X_train, Y_train, X_val, Y_val, X_test, Y_test = split_data(X,Y)

write_csv_file(filename="val_sv",X=X_val,Y=Y_val,subfolder="val")
write_csv_file(filename="test_sv",X=X_test,Y=Y_test,subfolder="test")
write_csv_file(filename="train_sv_100",X=X_train,Y=Y_train,subfolder="train")

#Split the data randomly with different data sizes
X_train_25, Y_train_25 = split_randomly(X_train,Y_train,data_size=0.25)
X_train_50, Y_train_50 = split_randomly(X_train,Y_train,data_size=0.50)
X_train_75, Y_train_75 = split_randomly(X_train,Y_train,data_size=0.75)

#Write to files
write_csv_file(filename="train_sv_25",X=X_train_25,Y=Y_train_25,subfolder="train")
write_csv_file(filename="train_sv_50",X=X_train_50,Y=Y_train_50,subfolder="train")
write_csv_file(filename="train_sv_75",X=X_train_75,Y=Y_train_75,subfolder="train")

#Split the data
X_train, Y_train, X_val, Y_val, X_test, Y_test = split_data(X,Y)

write_csv_file(filename="val_sv",X=X_val,Y=Y_val,subfolder="val")
write_csv_file(filename="test_sv",X=X_test,Y=Y_test,subfolder="test")
write_csv_file(filename="train_sv_100",X=X_train,Y=Y_train,subfolder="train")

#Split the data randomly with different data sizes
X_train_10, Y_train_10 = split_randomly(X_train,Y_train,data_size=0.10)
X_train_25, Y_train_25 = split_randomly(X_train,Y_train,data_size=0.25)
X_train_50, Y_train_50 = split_randomly(X_train,Y_train,data_size=0.50)
X_train_75, Y_train_75 = split_randomly(X_train,Y_train,data_size=0.75)

#Write to files
write_csv_file(filename="train_sv_10",X=X_train_10,Y=Y_train_10,subfolder="train")
write_csv_file(filename="train_sv_25",X=X_train_25,Y=Y_train_25,subfolder="train")
write_csv_file(filename="train_sv_50",X=X_train_50,Y=Y_train_50,subfolder="train")
write_csv_file(filename="train_sv_75",X=X_train_75,Y=Y_train_75,subfolder="train")

#Generate unique test data
generate_unique_test_data(lang='sv')


print("Creating english data...")

#Create english data
X_train_en, Y_train_en = translate_text_to_eng_batch(X_train,Y_train)
write_csv_file(filename="train_en_100",X=X_train_en,Y=Y_train_en,subfolder="train")

X_val_en, Y_val_en = translate_text_to_eng_batch(X_val,Y_val)
write_csv_file(filename="val_en",X=X_val_en,Y=Y_val_en,subfolder="val")

X_test_en, Y_test_en = translate_text_to_eng_batch(X_test,Y_test)
write_csv_file(filename="test_en",X=X_test_en,Y=Y_test_en,subfolder="test")

X_train_en_10, Y_train_en_10 = translate_text_to_eng_batch(X_train_10,Y_train_10)
write_csv_file(filename="train_en_10",X=X_train_en_10,Y=Y_train_en_10,subfolder="train")

X_train_en_25, Y_train_en_25 = translate_text_to_eng_batch(X_train_25,Y_train_25)
write_csv_file(filename="train_en_25",X=X_train_en_25,Y=Y_train_en_25,subfolder="train")

X_train_en_50, Y_train_en_50 = translate_text_to_eng_batch(X_train_50,Y_train_50)
write_csv_file(filename="train_en_50",X=X_train_en_50,Y=Y_train_en_50,subfolder="train")

X_train_en_75, Y_train_en_75 = translate_text_to_eng_batch(X_train_75,Y_train_75)
write_csv_file(filename="train_en_75",X=X_train_en_75,Y=Y_train_en_75,subfolder="train")

generate_unique_test_data(lang="en")