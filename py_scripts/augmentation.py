# Functions for data augmentation
import numpy as np
import random
from translation import translate_text_to_eng_batch, translate_text_to_swe_batch
from file_handler import read_csv_file, write_csv_file


def back_translation(X,Y):
    #Back translate the text
    X_new, Y_new = translate_text_to_eng_batch(X,Y)
    X_new, Y_new = translate_text_to_swe_batch(X_new,Y_new)

    #Write the new data to a csv file
    write_csv_file("back_translation", X_new, Y_new, "augmented")
        
    return X_new, Y_new
