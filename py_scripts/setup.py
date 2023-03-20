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

X, Y = preprocessing(IOB=True,punctuation=punctuation,cased=False)
X_cased, Y_cased = preprocessing(IOB=True,punctuation=punctuation,cased=True)

#Create the data directories
create_data_dirs()

#Write to processed data
write_csv_file(filename="cleaned",X=X,Y=Y,subfolder="processed")
write_csv_file(filename="cleaned_cased",X=X_cased,Y=Y_cased,subfolder="processed")

#Create all the uncased data
#Split the data
X_train, Y_train, X_val, Y_val, X_test, Y_test = split_data(X,Y)

write_csv_file(filename="val_swe",X=X_val,Y=Y_val,subfolder="val")
write_csv_file(filename="test_swe",X=X_test,Y=Y_test,subfolder="test")
write_csv_file(filename="train_swe_100",X=X_train,Y=Y_train,subfolder="train")

#Split the data randomly with different data sizes
X_train_25, Y_train_25 = split_randomly(X_train,Y_train,data_size=0.25)
X_train_50, Y_train_50 = split_randomly(X_train,Y_train,data_size=0.50)
X_train_75, Y_train_75 = split_randomly(X_train,Y_train,data_size=0.75)

#Write to files
write_csv_file(filename="train_swe_25",X=X_train_25,Y=Y_train_25,subfolder="train")
write_csv_file(filename="train_swe_50",X=X_train_50,Y=Y_train_50,subfolder="train")
write_csv_file(filename="train_swe_75",X=X_train_75,Y=Y_train_75,subfolder="train")

#Create all the cased data
#Split the data
X_train_cased, Y_train_cased, X_val_cased, Y_val_cased, X_test_cased, Y_test_cased = split_data(X_cased,Y_cased)

write_csv_file(filename="val_swe_cased",X=X_val_cased,Y=Y_val_cased,subfolder="val")
write_csv_file(filename="test_swe_cased",X=X_test_cased,Y=Y_test_cased,subfolder="test")
write_csv_file(filename="train_swe_100_cased",X=X_train_cased,Y=Y_train_cased,subfolder="train")

#Split the data randomly with different data sizes
X_train_10_cased, Y_train_10_cased = split_randomly(X_train_cased,Y_train_cased,data_size=0.10)
X_train_25_cased, Y_train_25_cased = split_randomly(X_train_cased,Y_train_cased,data_size=0.25)
X_train_50_cased, Y_train_50_cased = split_randomly(X_train_cased,Y_train_cased,data_size=0.50)
X_train_75_cased, Y_train_75_cased = split_randomly(X_train_cased,Y_train_cased,data_size=0.75)

#Write to files
write_csv_file(filename="train_swe_10_cased",X=X_train_10_cased,Y=Y_train_10_cased,subfolder="train")
write_csv_file(filename="train_swe_25_cased",X=X_train_25_cased,Y=Y_train_25_cased,subfolder="train")
write_csv_file(filename="train_swe_50_cased",X=X_train_50_cased,Y=Y_train_50_cased,subfolder="train")
write_csv_file(filename="train_swe_75_cased",X=X_train_75_cased,Y=Y_train_75_cased,subfolder="train")

#Generate unique test data
generate_unique_test_data(uncased=True)
generate_unique_test_data(uncased=False)


print("Creating english data...")

#Create english data (uncased)
X_train_en, Y_train_en = translate_text_to_eng_batch(X_train_cased,Y_train_cased)
X_val_en, Y_val_en = translate_text_to_eng_batch(X_val_cased,Y_val_cased)
X_test_en, Y_test_en = translate_text_to_eng_batch(X_test_cased,Y_test_cased)

X_train_en_25, Y_train_en_25 = translate_text_to_eng_batch(X_train_25_cased,Y_train_25_cased)
X_train_en_50, Y_train_en_50 = translate_text_to_eng_batch(X_train_50_cased,Y_train_50_cased)
X_train_en_75, Y_train_en_75 = translate_text_to_eng_batch(X_train_75_cased,Y_train_75_cased)

#Write to files
write_csv_file(filename="train_eng_100_cased",X=X_train_en,Y=Y_train_en,subfolder="train")
write_csv_file(filename="val_eng_cased",X=X_val_en,Y=Y_val_en,subfolder="val")
write_csv_file(filename="test_eng_cased",X=X_test_en,Y=Y_test_en,subfolder="test")

write_csv_file(filename="train_eng_25_cased",X=X_train_en_25,Y=Y_train_en_25,subfolder="train")
write_csv_file(filename="train_eng_50_cased",X=X_train_en_50,Y=Y_train_en_50,subfolder="train")
write_csv_file(filename="train_eng_75_cased",X=X_train_en_75,Y=Y_train_en_75,subfolder="train")

