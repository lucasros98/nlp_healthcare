import os
import sys
import pandas as pd
from torch import nn
from dotenv import load_dotenv, find_dotenv
sys.path.append(os.path.dirname(find_dotenv()))
load_dotenv(find_dotenv())

from transformers import AutoTokenizer, AutoModel
from py_scripts.file_handler import save_result_file, read_csv_file
from py_scripts.data import get_training_data, get_unique_test, build_file_name, get_augmented_data
import py_scripts.ner_util.ner_system as ner_util
import py_scripts.ner_util.evaluation as evaluation
from parameters import NERParameters


def build_params_list(combinations):
    results = {}
    for comb in combinations:
        #loop of keys and values
        for key, value in comb.items():
            if key not in results:
                results[key] = []
            results[key].append(str(value))

    #remove all duplicates -> set variable if only one value
    aug_params = {}
    aug_params['aug_method'] = list(set(results['aug_method'])) if len(set(results['aug_method'])) > 1 else results['aug_method'][0]
    aug_params['p'] = list(set(results['p'])) if len(set(results['p'])) > 1 else results['p'][0]
    aug_params['num_new_docs'] = list(set(results['num_new_docs'])) if len(set(results['num_new_docs'])) > 1 else results['num_new_docs'][0]
    aug_params['data_size'] = list(set(results['data_size'])) if len(set(results['data_size'])) > 1 else results['data_size'][0]
    aug_params['bt_type'] = list(set(results['bt_type'])) if len(set(results['bt_type'])) > 1 else results['bt_type'][0]
    
    return aug_params


def run_model(model_name="kb_bert",bert_model="KB/bert-base-swedish-cased",local_files_only=False,add_prefix_space=False,precentage=100,lang='sv',runs=5,aug_params_list=None, aug_combination=''):

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
    X_train,Y_train,X_val,Y_val,X_test,Y_test = get_training_data(precentage=precentage,lang=lang)
    X_test_unique, Y_test_unique = get_unique_test(lang=lang)
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

    if aug_params_list is not None:
        aug_run_tag = ""
        for aug_params in aug_params_list:
            file_name = build_file_name(**aug_params)
            aug_run_tag += file_name + "_"

            subfolder = "augmented" if not isinstance(aug_params['aug_method'], list) else "augmented_merged"
            X_aug, Y_aug = read_csv_file(file_name + '.csv', subfolder=subfolder)
            X_train += X_aug
            Y_train += Y_aug

        print("Augmented data len:", len(X_train))
        params.aug_params = build_params_list(aug_params_list)
        params.run_tag = f"{int(precentage)}_{aug_combination}_{aug_run_tag}"

    #Run the model 5 or less times with different random seeds
    for i in range(runs):
        params.random_seed = i
        params.run_name = params.run_tag + "_" + str(i)
        #Instantiate the NER system
        ner_system = ner_util.SequenceLabeler(params, Model, bert_tokenizer=AutoTokenizer.from_pretrained(bert_model,local_files_only=local_files_only,add_prefix_space=add_prefix_space))

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
        filename = model_name + "_" + str(int(precentage)) + "_" + aug_params['aug_method'] + "_n" + str(aug_params['num_new_docs']) + ".csv"
        save_result_file(model_name,filename, best_results)
    except:
        print("Error occured while saving the results. Please check the sys args.")

    #Evaluation on some examples
    evaluation.print_examples(ner_system, lang)
