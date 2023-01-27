# NLP for HealthCare - Master Thesis

This repository contains the code for our master thesis. The thesis is about using NER to extract information from medical records.

## Project structure

The project is structured as follows:

- `data`: contains the data used for training and testing the models
- `notebooks_exploration_cleaning`: contains the notebooks used for data exploration, preprocessing and cleaning
- `notebooks_models`: contains the notebooks used for training and testing the models
- `py_scripts`: contains the general python scripts such as the visualization script

## Data

The data used for training and testing the models is not included in this repository because of privacy concerns.

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
