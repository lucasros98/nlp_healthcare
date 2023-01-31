import pandas as pd

def read_csv(path):
    df = pd.read_csv(path)
    X = df["X"].tolist()
    Y = df["Y"].tolist()
    return X, Y

def read_csv_file(filename):
    return read_csv("../data/" + filename)