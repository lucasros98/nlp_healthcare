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

#Defining the model
class Model(nn.Module):
    def __init__(self, seq_labeler):
        super().__init__()
        self.bert = AutoModel.from_pretrained('/mimer/NOBACKUP/groups/snic2021-23-309/project-data/nlp-for-healthcare/SweDeClin-BERT_copy/',local_files_only=True)
        self.top_layer = nn.Linear(self.bert.config.hidden_size, seq_labeler.n_labels)

    def forward(self, words):
        outputs = self.bert(words)
        res = outputs[0]
        return self.top_layer(res)

try:
    precentage = float(float(sys.argv[1])) if len(sys.argv) > 1 and sys.argv[1] != "None" else 100
except:
    precentage = 100
    print("Error occured while parsing the precentage from the sys args. Please check the sys args. Using {}% of the data.".format(precentage))


#Loading the data
X_train,Y_train,X_val,Y_val,X_test,Y_test = get_training_data(precentage=precentage,uncased=False)
X_test_unique, Y_test_unique = get_unique_test()
print(f"Length of the data:\nTrain: {len(X_train)}\nValidation: {len(X_val)}\nTest: {len(X_test)}")

# Finetuning BERT model
params = NERParameters()
results = []
best_results = pd.DataFrame()

#Set model name
curr_file = os.path.basename(__file__).split(".")[0]
params.model_name = curr_file + "_{}".format(str(int(precentage)))

#Run the model with 5 times with different random seeds
for i in range(5):
    params.random_seed = i
    params.run_name = params.model_name + "_{}".format(i)

    #Instantiate the NER system
    ner_system = ner_util.SequenceLabeler(params, Model, bert_tokenizer=AutoTokenizer.from_pretrained('/mimer/NOBACKUP/groups/snic2021-23-309/project-data/nlp-for-healthcare/SweDeClin-BERT_copy/',local_files_only=True))

    #Fit the model
    ner_system.fit(X_train, Y_train, X_val, Y_val)

    #Evaluation of the system
    res = ner_system.evaluate_model(X_test,Y_test)
    results.append(res)

    #Evaluation on unique test data
    if len(X_test_unique) > 0:
        res_unique = ner_system.evaluate_model(X_test_unique,Y_test_unique)

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
    filename = curr_file + "_" + str(int(precentage)) + ".csv"
    save_result_file(curr_file,filename, best_results)
except:
    print("Error occured while saving the results. Please check the sys args.")

#Evaluation on some examples
evaluation.print_examples(ner_system, 'sv')