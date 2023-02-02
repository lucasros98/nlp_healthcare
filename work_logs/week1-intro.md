# Week 1: Introduction

The first week of the project consisted out of the following tasks:
 - Completing the introduction to Alvis and computer clusters
 - Preprocessing of the data
 - Exploration of the data
 - Setting up the project structure, repository and master thesis document in Overleaf

## Exploration of the data

The data exploration was done in the notebook `notebooks_exploration_cleaning/exploration.ipynb`.

Some interesting foundings were:
- a few instances of abbriviations being used for words in normal language, such as:
  - "hö" instead of "höger"
  - "vä" instead of "vänster"
  - "fr" instead of "från"
  - "pat" instead of "patient"
  - "beh" instead of "behandlas"
  - "bed" instead of "bedömning" (?)
  - "avd" instead of "avdelning"
  - "bakt" instead of "bakterier / bakterie"
  - "rel" instead of "relativt"
  - "perm" instead of "permission"
  - "mkt" instead of "mycket"
  - "stud" instead of "student / studerande"
  - "vb" instead of "vid behov" (?)
  - "ang" instead of "angående"
  - "enl" instead of "enligt"
  - "enh" instead of "enhet" (?)

## Data cleaning and preprocessing

The data cleaning and preprocessing was done in the notebook `notebooks_exploration_cleaning/cleaning.ipynb`.

At this moment we are not sure if we will use IO-tagging or IOB-tagging. We will have to look into this more.

The reason for this is because the data has many IOB-tags. For example for the entity "FULL_DATE", which we believe could be important for the models.

However, other entities has just a very few IOB-tags.

There are also some punctuations that we believe could be important for the models, such as "->" or "-" between dates. We will have to look into this more. 