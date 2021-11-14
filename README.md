# Automatic Topic Labeling with Thesaurus-based methods

This repo implements the strategies outlined in *Principled Analysis of Energy Discourse across Domains with Thesaurus-based Automatic Topic Labeling (2021)* and is an easy way to implement an automatic labelling technique for LDA-style topics or lists of keywords using a pre-defined thesaurus.

If any questions arise please don't hesitate to contact me at:

tscelsi@student.unimelb.edu.au

## Installation Instructions

#### 1. Clone repository
#### 2. Install as package in a python environment
```
pip install -e dtm_toolkit
```
#### 3. Use in python scripts
```
from dtm_toolkit.auto_labelling import AutoLabel
...
```

## Contents

#### Examples
In the examples folder ```dtm_toolkit/examples```, we show some examples of analying using the dtm analysis module of the toolkit, and also some simple examples of the automatic topic labelling procedure.

#### DTM Toolkit

Within the toolkit we mainly provide tools for preprocessing text, automatic labelling and the creation of valid input for the dynamic topic model (DTM) as implemented [here](https://github.com/blei-lab/dtm).