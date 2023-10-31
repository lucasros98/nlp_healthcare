# Master Thesis - Enhancing Named Entity Recognition Performance using Data Augmentation

**The paper can be found here**: https://gupea.ub.gu.se/handle/2077/78677

This repository contains the code for our master thesis that explored how data augmentation could be used to improve the performance of NER models for Swedish clinical text. The dataset used for this project is the Stockholm EPR PHI Pseudo Corpus.

In this study we explored and compared the performance of five different models: **KB-BERT, M-BERT, SweDeClin-BERT, Bio-BERT and Clinical BERT**. We also explored nine different data augmentation techniques on the best-performing model, SweDeClin-BERT.

The data augmentation techniques that were explored were:

- Back-Translation using word for word translation
- Back-Translation using sense for sense translation
- Label-wise Token Replacement
- Synonym Replacement
- Mention Replacement with new entities
- Mention Replacement with original entities
- Shuffle within Segments
- Random Deletion
- BERT-Mask Replacement

## Install requirements with pip

To install the requirements, you need to run the following command:

`$ pip install -r requirements.txt`

## Project structure

The project is structured as follows:

- `data`: contains the data used for training and testing the models. The data is not included in this repository because of privacy concerns.
- `data_public`: contains the public data used for generating new data using data augmentation.
- `notebooks_exploration_cleaning`: contains the notebooks used for data exploration, preprocessing and cleaning
- `notebooks_models`: contains the notebooks used for training and testing the models
- `py_scripts`: contains the general python scripts such as the visualization script
- `setup_scripts`: contains scripts used for generating the data used for training and testing the models.

## Environment variables

The project uses environment variables to store the paths to the data and models. The environment variables are:

- `DATA_PATH`: path to the original data (txt file)
- `MODEL_PATH`: path to the folder where the models are stored
- `DATA_DIR`: path to the folder where all the data is stored
- `PUBLIC_DATA_DIR` path to the folder where the public data is stored
- `RESULT_DIR`: path to the folder where the results are stored after training and testing the models.

Before running any notebooks or files, you need to set these environment variables. This can be done by creating a `.env` file in the root of the project and adding the environment variables to the file. The `.env` file should look like this:

```bash
    DATA_PATH = path/to/data.txt
    MODEL_PATH = path/to/models
    DATA_DIR = path/to/data_dir
    PUBLIC_DATA_DIR = path/to/public_data_dir
    RESULT_DIR = path/to/result_dir
```

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

To generate the train, validation and test data, you need to run the setup.py file in the setup_scripts folder.

```bash
    python3 setup.py
```

This script will generate the train, validation and test data in the data folder. Both for sv and en used for training and testing the models.

To translate the text into English we use the MarianMT model from HuggingFace. This model is downloaded automatically when running the setup.py script.

## Setup scripts

The folder `setup_scripts` contains the scripts used for generating the data used for training and testing the models.

The folder contains three scripts:

- `setup.py`: generates the train, validation and test data from a txt file containing the data.
- `translation_setup.py`: translates the data into English and Swedish.
- `augemntation_setup.py`: generates the augmented data using training data and public data.

Before any augmentation can be done, the training data needs to be generated using the `setup.py` script.


## Models

The models that was compared in our study were **KB-BERT, M-BERT, SweDeClin-BERT, Bio-BERT and Clinical BERT**.

KB-BERT, M-BERT, Bio-BERT and Clinical BERT can be downloaded from HuggingFace, while SweDeClin-BERT was obtained from the Swedish Health Record Research Bank.

We also developed two simple baselines. The first model only predicts the most frequently occurring entity of each class, while the second model uses a dictionary to predict entities based on a list of known entities. The code for these can be found in the notebooks_models folder.


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

The most important scripts are:

- `augmentation.py`: contains the code for the different data augmentation techniques.
- `generation.py`: contains the code for generating unique entities.
- `translation.py`: contains the code for translating the text into English and Swedish.
- `data.py`: contains a lot of helper functions used for handling and processing the data.
- `model_training.py`: contains the code for running and training the models.
