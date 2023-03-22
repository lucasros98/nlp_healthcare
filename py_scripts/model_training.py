import os
import sys
import pandas as pd
from torch import nn
from dotenv import load_dotenv, find_dotenv
sys.path.append(os.path.dirname(find_dotenv()))
load_dotenv(find_dotenv())

from transformers import AutoTokenizer, AutoModel
from py_scripts.file_handler import save_result_file
from py_scripts.data import get_training_data, get_unique_test
import py_scripts.ner_util.ner_system as ner_util
import py_scripts.ner_util.evaluation as evaluation
from parameters import NERParameters


def run_model(model_name="kb_bert",bert_model="KB/bert-base-swedish-cased",local_files_only=False,precentage=100,lang='sv',runs=5,aug_params=None):

    #Defining the model
    class Model(nn.Module):
        def __init__(self, seq_labeler):
            super().__init__()
            self.bert = AutoModel.from_pretrained(bert_model,local_files_only=local_files_only)
            self.top_layer = nn.Linear(self.bert.config.hidden_size, seq_labeler.n_labels)

        def forward(self, words):
            outputs = self.bert(words)
            res = outputs[0]
            return self.top_layer(res)

    #Loading the data
    X_train,Y_train,X_val,Y_val,X_test,Y_test = get_training_data(precentage=precentage,uncased=False)
    X_test_unique, Y_test_unique = get_unique_test()
    print(f"Length of the data:\nTrain: {len(X_train)}\nValidation: {len(X_val)}\nTest: {len(X_test)}")

    # Instantiate the parameters
    params = NERParameters()
    results = []
    best_results = pd.DataFrame()

    #Set model name and other parameters
    curr_file = os.path.basename(__file__).split(".")[0]
    params.model_name = model_name
    params.data_size = precentage
    params.run_tag = str(int(precentage)) + "_normal" 

    if aug_params is not None:
        params.run_tag = str(int(precentage)) + "_augmented"
        params.aug_params = aug_params

    #Run the model with 5 times with different random seeds
    for i in range(runs):
        params.random_seed = i
        params.run_name = params.run_tag + "_" + str(i)
        #Instantiate the NER system
        ner_system = ner_util.SequenceLabeler(params, Model, bert_tokenizer=AutoTokenizer.from_pretrained(bert_model,local_files_only=local_files_only))

        #Fit the model
        ner_system.fit(X_train, Y_train, X_val, Y_val)

        #Evaluation of the system
        res = ner_system.evaluate_model(X_test,Y_test)
        results.append(res)

        #Evaluation of the system on unique test data
        if len(X_test_unique) > 0:
            res_unique = ner_system.evaluate_model(X_test_unique,Y_test_unique,unique_labels=True)

        #Save the best results
        if best_results.empty:
            best_results = res
        else:
            overall_f1 = res.loc[res['entity'] == 'overall', 'f1'].values[0]
            best_f1 = best_results.loc[best_results['entity'] == 'overall', 'f1'].values[0]

            if overall_f1 > best_f1:
                best_results = res

    average_results = evaluation.calculate_average_results(results)

    print("Average results:")
    avg_df = pd.DataFrame.from_dict(average_results, orient='index')
    print(avg_df)    

    #Create a file name based on the script name and the precentage of the data used for training
    #Save the results to file
    try:
        filename = model_name + "_" + str(int(precentage)) + ".csv"
        save_result_file(model_name,filename, best_results)
    except:
        print("Error occured while saving the results. Please check the sys args.")

    #Evaluation on some examples
    evaluation.print_examples(ner_system, lang)