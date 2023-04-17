# Description: This script performs a significance test on the results of a set of experiments.

from scipy.stats import f_oneway
import csv
import os
import sys
import pandas as pd

from dotenv import load_dotenv,find_dotenv

sys.path.append(os.path.dirname(find_dotenv()))
load_dotenv(find_dotenv())


def significance_test(results=[]):
    """
    Performs a significance test on the results of a set of experiments.
    The results should be a list of lists, where each list contains the
    results of a single experiment.
    """
    #check that there are at least two results
    if len(results) < 2:
        return False
    
    #check that all results have the same number of elements
    num_elements = len(results[0])
    for result in results:
        if len(result) != num_elements:
            return False
    
    #perform the significance test
    f, p = f_oneway(*results)

    #return the result
    return p

def load_data():
    result_dir = os.environ.get("RESULT_DIR")
    if result_dir is None:
        print("Please set the RESULT_DIR environment variable.")
        return 
    
    path = os.path.join(result_dir, "results_augmented")

    results = []
    for filename in os.listdir(path):
        if filename.endswith(".csv"):
            results.append(load_csv_file(os.path.join(path, filename)))

    return results

def load_csv_file(filename):
    try:
        data = pd.read_csv(filename)
        
        #convert the data to a list for f1 column
        data = data['f1'].tolist()
        return data
    except Exception as e:
        print(f"Error occurred while loading CSV file: {e}. Please check the RESULT_DIR environment variable and file name.")


print(significance_test(load_data()))