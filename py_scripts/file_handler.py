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
        X = [ast.literal_eval(x) for x in X]
        Y = [ast.literal_eval(y) for y in Y]
        return X, Y
    except Exception as e:
        print(f"Error reading CSV file: {e}. Please check the path and file name.")
        return None, None

#read csv file given a filename from data folder
def read_csv_file(filename,subfolder=''):
    if subfolder:
        subfolder += "/"

    data_dir = os.environ.get("DATA_DIR")
    if data_dir is None:
        print("Please set the DATA_DIR environment variable.")
        return None, None
    
    filepath = os.path.join(data_dir, subfolder, filename)
    return read_csv(filepath)

#read csv file given a filename from public data folder
def read_public_csv(filename,delimiter=';'):
    public_data_dir = os.environ.get("PUBLIC_DATA_DIR")

    if public_data_dir is None:
        print("Please set the PUBLIC_DATA_DIR environment variable.")
        return {}
    try:
        with open(os.path.join(public_data_dir, filename), 'r') as file:
            dict = {}
            for line in file:
                split = line.split(delimiter)

                key = split[0].lower()
                value = split[1].lower()

                #Remove newline character (\n)
                value = value[:-1]

                dict[key] = value
            return dict
    except Exception as e:
        print(f"Error reading public CSV file: {e}.")
        return {}

#write csv file given a filename and X and Y
def write_csv_file(filename, X, Y, subfolder=""):
    if subfolder:
        subfolder += "/"
    
    data_dir = os.environ.get("DATA_DIR")
    if data_dir is None:
        print("Please set the DATA_DIR environment variable.")
        return

    try:
        result = pd.DataFrame({"X": X, "Y": Y})
        result.to_csv(os.path.join(data_dir, subfolder, f"{filename}.csv"), index=False)
    except Exception as e:
        print(f"Error occurred while creating CSV file: {e}. Please check the DATA_DIR environment variable and file name.")


def create_dir(path):
    if not os.path.exists(path):
        print("Creating a new dir for saving results..")
        os.makedirs(path, exist_ok=True)

def save_result_file(subfolder, filename, result):
    result_dir = os.environ.get("RESULT_DIR")
    if result_dir is None:
        print("Please set the RESULT_DIR environment variable.")
        return

    path = os.path.join(result_dir, subfolder)
    create_dir(path)
    filepath = os.path.join(path, filename)

    try:
        print(f"Saving to {filepath}")
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
        print(f"Error occurred while creating CSV file: {e}. Please check the RESULT_DIR environment variable and file name.")