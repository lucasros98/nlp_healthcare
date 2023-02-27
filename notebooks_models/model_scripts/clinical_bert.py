#!/usr/bin/env python
# coding: utf-8

# # Bio-BERT

# In[ ]:


import os
from dotenv import load_dotenv, find_dotenv
import sys
from torch import nn

sys.path.append(os.path.dirname(find_dotenv()))
load_dotenv(find_dotenv())


# In[ ]:


from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
bert_model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')


# ## Getting the data

# In[ ]:


#Import the file_handler
from py_scripts.file_handler import read_csv_file,save_result_file

from py_scripts.data import print_unknown_tokens, split_randomly

#Import the NER system
import py_scripts.ner_util.ner_system as ner_util

#Import evaluation functions
import py_scripts.ner_util.evaluation as evaluation


# In[ ]:


#Load data 
X, Y = read_csv_file("translated.csv")


# ## Exploring the BERT tokenizer on clincial text

# In[ ]:


#Explore the tokenizer by finding all the unknown tokens in the data and printing them
print_unknown_tokens(tokenizer, X)


# ## Preparing the data

# In[ ]:


from sklearn.model_selection import train_test_split

#Ratio of train, validation and test
train_ratio = 0.8
validation_ratio = 0.10
test_ratio = 0.10

#Random state - For reproducibility
random_state=104

#Split data into train, validation and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1-train_ratio, random_state=random_state)
X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=test_ratio/(test_ratio+validation_ratio), random_state=random_state)

#Get the precentage of the data that should be used for training
try:
    precentage = float(float(sys.argv[1])/100) if len(sys.argv) > 1 and sys.argv[1] != "None" else 1.0
except:
    print("Error occured while parsing the precentage from the sys args. Please check the sys args.")
    precentage = 1.0

X_train, Y_train = split_randomly(X_train, Y_train, precentage)

print("Using " + str(precentage*100) + "% of the data for training.")


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

#Update the parameters if needed
params.tagging_scheme = "BIO"


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
    filename = curr_file + "_" + str(int(precentage*100)) + ".csv"
    save_result_file(curr_file,filename, res)
except:
    print("Error occured while saving the results. Please check the sys args.")

evaluation.print_examples(ner_system, 'en')

