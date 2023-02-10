# Week 2

## Premilinary plan for the week

- Implement a simple baseline (predicting the most common tag for each word)
- Start on a NER system, vocabulary etc.
- Think about structure for the thesis (what to write about, what to include)
- Writing on the thesis 
  - Word embeddings
  - Transformers
  - BERT
- Continue to learn about Alvis and running jobs on the cluster

### From the monday meeting with supervisor:

- (Ta fram en lista över förekommande förkortningar.)
- Olika typer av kategorier -> (sammanfatta och ta fram en lista över kategorier)
- Kolla om "->" finns med i BERTs vocab -> **Svar ja**

## Completed tasks

For this week we have completed the following tasks:

### Implemented simple baselines

We implemented two simple baselines, one that predicts the most common tag for each word and one that uses a dictionary to predict the tag. The dictionary is used for matching words in the training data that we have seen before.

### Writing on the thesis

This week we also did some writing on the thesis. We wrote about:
- Transformers
- BERT
  
We also tried to create a structure for the thesis.

### Translation system

We also started to work on a translation system that we will use later to translate the Swedish data to English (with the MarianMT model from HuggingFace).

We did this to test that we could use the MarianMT model and to see if we could get a good translation.

One negative thing is that it takes a long time to translate all the data. 

### Started on a NER system, vocabulary etc.

We also started to work on a NER system that we will use later to train the different BERT models. We have a little bit more work to do on this, but we have a good start.


## Thoughts?


