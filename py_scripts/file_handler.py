import pandas as pd
import ast

#read csv file given a path and return X and Y
def read_csv(path):
    df = pd.read_csv(path)
    X = df["X"].tolist()
    Y = df["Y"].tolist()

    #convert strings into lists with map function
    X = list(map(lambda x: ast.literal_eval(x), X))
    Y = list(map(lambda x: ast.literal_eval(x), Y))
    
    return X, Y

#read csv file given a filename from data folder
def read_csv_file(filename):
    return read_csv("../data/" + filename)

#write csv file given a filename and X and Y
def write_csv_file(filename, X, Y):
    result = pd.DataFrame({"X": X, "Y": Y})
    result.to_csv("../data/"+filename+".csv")