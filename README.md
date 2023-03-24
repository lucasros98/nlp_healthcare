# NLP for HealthCare - Master Thesis

This repository contains the code for our master thesis. The thesis is about using NER to extract information from Swedish clinical records.

The dataset used for this project is the Stockholm EPR PHI Pseudo Corpus.

## Install requirements with pip
`$ pip install -r requirements.txt`

## Project structure

The project is structured as follows:

- `data`: contains the data used for training and testing the models
- `data_public`: contains the public data used for generating new data (data augmentation)
- `models`: contains the models used for training and testing
- `notebooks_exploration_cleaning`: contains the notebooks used for data exploration, preprocessing and cleaning
- `notebooks_models`: contains the notebooks used for training and testing the models
- `py_scripts`: contains the general python scripts such as the visualization script

## Data

The data used for training and testing the models is not included in this repository because of privacy concerns.

How to guraantee that the data is not shared is explained in the next section (Using notebooks with cencored data).

### Using notebooks with cencored data

Since we are using notebooks with censored data, you can use the following command:

```bash
    pip install pre-commit
```

Then you need to write the following command in the terminal:

```bash
    pre-commit install
```

This will install the pre-commit hook that will automatically remove all outputs and prints from the notebooks.

### Generating the train, validation and test data

To generate the train, validation and test data, you need to run the setup.py file in the py_scripts folder.

```bash
    python3 setup.py
```

This script will generate the train, validation and test data in the data folder. Both for sv and en used for training and testing the models.

## Models

The models that will be compared in our study are **KB-BERT, M-BERT, SweDeClin-BERT, Bio-BERT and Clinical BERT**.

Additionally, as two simple baselines. The first model only predicts the most frequently occurring entity of each class, while the second model uses a dictionary to predict entities based on a list of known entities. 

To convert all ipynb to py in models folder, use the following command:

```bash

    ml load IPython/8.5.0-GCCcore-11.3.0

   jupyter nbconvert --output-dir='./model_scripts'  --to script *.ipynb
```

## Notebooks for data exploration, preprocessing and cleaning

The notebooks in this folder are used for data exploration, preprocessing and cleaning the data.

The name of the folder is `notebooks_exploration_cleaning` and the notebooks are:

- `exploration.ipynb`: contains the initial exploration for the data
- `cleaning.ipynb`: contains the preprocessing and cleaning of the data

## Notebooks for training and testing the models

The notebooks in this folder are used for training and testing the models.

The name of the folder is `notebooks_models`.

## Python scripts

This folder contains the python scripts used for multiple notebooks in order to avoid code duplication.

The name of the folder is `py_scripts`.