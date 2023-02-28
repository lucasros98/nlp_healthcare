import pandas as pd
import ast
import csv
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
def read_csv_file(filename,subfolder=''):
    if len(subfolder) > 0:
        subfolder = subfolder + "/"

    if(os.environ.get("DATA_DIR") == None):
        print("Please set the DATA_DIR environment variable.")
        return None, None
    
    filepath = os.environ.get("DATA_DIR") + subfolder + filename
    return read_csv(filepath)

#read csv file given a filename from public data folder
def read_public_csv(filename):
    if(os.environ.get("PUBLIC_DATA_DIR") == None):
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
def write_csv_file(filename, X, Y, subfolder=""):
    if len(subfolder) > 0:
        subfolder = subfolder + "/"
    try:
        result = pd.DataFrame({"X": X, "Y": Y})
        result.to_csv(os.environ.get("DATA_DIR")+subfolder+filename+".csv")
    except:
        print("Error occured while creating csv file. Please check the enviorment variables DATA_DIR and file name.")


def create_dir(path):
    if not os.path.exists(path):
        print("Creating a new dir for saving results..")
        os.makedirs(path, exist_ok=True)

def save_result_file(subfolder, filename, result):
    if(os.environ.get("RESULT_DIR") == None):
        print("Please set the RESULT_DIR environment variable.")
        return

    path = os.environ.get("RESULT_DIR") + subfolder + "/"

    #Try to create the folder if it doesn't exist
    create_dir(path)

    #create file path
    filepath = path+filename

    try:
        print("Saving to ", filepath)
        if not result.empty:
            #save to file using csv writer
            with open(filepath, 'w') as file:
                writer = csv.writer(file)
                writer.writerow(["entity", "precision", "recall", "f1", "number"])
                for index, row in result.iterrows():
                    writer.writerow([row["entity"], row["precision"], row["recall"], row["f1"], row["number"]])
        else:
            print("Result is empty")

    except Exception as e:
        print("Error occured while creating csv file. Please check the enviorment variable RESULT_DIR and file name.")
        print(e)