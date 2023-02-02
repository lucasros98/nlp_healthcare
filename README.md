# NLP for HealthCare - Master Thesis

This repository contains the code for our master thesis. The thesis is about using NER to extract information from Swedish clinical records.

The dataset used for this project is the Stockholm EPR PHI Pseudo Corpus.

## Project structure

The project is structured as follows:

- `data`: contains the data used for training and testing the models
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

## Models

The models that will be compared in our study are **KB-BERT, M-BERT, SweDeClin-BERT, and Bio-BERT**.

Additionally, as a simple baseline, we will implement a model that only will predict the most frequently occurring entity of each class.

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