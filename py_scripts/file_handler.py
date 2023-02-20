import pandas as pd
import ast
import os

#read csv file given a path and return X and Y
def read_csv(path):
    try:
        df = pd.read_csv(path)
        X = df["X"].tolist()
        Y = df["Y"].tolist()

        #convert strings into lists with map function
        X = list(map(lambda x: ast.literal_eval(x), X))
        Y = list(map(lambda x: ast.literal_eval(x), Y))
        return X, Y
    except:
        print("Error reading csv file. Please check the path and file name.")
        return None, None

#read csv file given a filename from data folder
def read_csv_file(filename):
    if(os.environ.get("DATA_DIR") == None):
        print("Please set the DATA_DIR environment variable.")
        return None, None
    return read_csv(os.environ.get("DATA_DIR") + filename)

#read csv file given a filename from public data folder
def read_public_csv(filename):
    if(os.environ.get("DATA_DIR") == None):
        print("Please set the PUBLIC_DATA_DIR environment variable.")
        return {}
    
    with open(os.environ.get("PUBLIC_DATA_DIR") + filename, 'r') as file:
        dict = {}
        for line in file:
            split = line.split(';')

            key = split[0].lower()
            value = split[1].lower()

            #Remove newline character (\n)
            value = value[:-1]

            dict[key] = value
    return dict

#write csv file given a filename and X and Y
def write_csv_file(filename, X, Y):
    try:
        result = pd.DataFrame({"X": X, "Y": Y})
        result.to_csv(os.environ.get("DATA_DIR")+filename+".csv")
    except:
        print("Error occured while creating csv file. Please check the enviorment variables DATA_DIR and file name.")