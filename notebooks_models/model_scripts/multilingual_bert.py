#!/usr/bin/env python
# coding: utf-8

# # Multilingual-BERT

# In[ ]:


import os
from dotenv import load_dotenv, find_dotenv
import sys
from torch import nn

sys.path.append(os.path.dirname(find_dotenv()))
load_dotenv(find_dotenv())


# In[ ]:


from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
bert_model = AutoModel.from_pretrained("bert-base-multilingual-cased")


# ## Getting the data

# In[ ]:


#Import the file_handler
from py_scripts.file_handler import save_result_file

from py_scripts.data import get_training_data

#Import the NER system
import py_scripts.ner_util.ner_system as ner_util

#Import evaluation functions
import py_scripts.ner_util.evaluation as evaluation


# In[ ]:


#Get precentage of data to use
try:
    precentage = float(float(sys.argv[1])) if len(sys.argv) > 1 and sys.argv[1] != "None" else 100
except:
    precentage = 100
    print("Error occured while parsing the precentage from the sys args. Please check the sys args. Using {}% of the data.".format(precentage))


# In[ ]:


#Load data 
X_train,Y_train,X_val,Y_val,X_test,Y_test = get_training_data(precentage=precentage)


# In[ ]:


#Print the length of the data
print("Length of the data:")
print("Train: " + str(len(X_train)))
print("Validation: " + str(len(X_val)))
print("Test: " + str(len(X_test)))


# ## Defining the model

# In[ ]:


class Model(nn.Module):
    def __init__(self, seq_labeler):
        super().__init__() 

        # BERT model.
        self.bert = bert_model

        # Output unit.
        self.top_layer = nn.Linear(self.bert.config.hidden_size, seq_labeler.n_labels)

    def forward(self, words):
        outputs = self.bert(words)
        res = outputs[0]
        return self.top_layer(res)


# ### Defining NER Parameters

# In[ ]:


#Import NER parameters from parameters.py
from parameters import NERParameters

params = NERParameters()


# ## Finetuning BERT model

# In[ ]:


ner_system = ner_util.SequenceLabeler(params, Model, bert_tokenizer=tokenizer)

ner_system.fit(X_train, Y_train, X_val, Y_val)


# ## Evaluation of the system
# 
# Evaluate the sytem on the test data.

# In[ ]:


res = ner_system.evaluate_model(X_test,Y_test)

#Create a file name based on the script name and the precentage of the data used for training
try:
    curr_file = os.path.basename(__file__).split(".")[0]
    filename = curr_file + "_" + str(int(precentage)) + ".csv"
    save_result_file(curr_file,filename, res)
except:
    print("Error occured while saving the results. Please check the sys args.")

evaluation.print_examples(ner_system, 'en')

