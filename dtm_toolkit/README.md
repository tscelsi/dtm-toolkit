# Main Modules

### ```auto_labelling.py```
This module contains all the logic for the automatic labelling of lists of keywords with pre-defined thesaurus labels. See the ```../examples/auto_labelling_examples.ipynb``` notebook for more details.
### ```preprocessing.py```
A simple module that contains logic for preprocessing of text for input into the dynamic topic model (DTM).
### ```dtm/creator.py```
Contains the logic for the creation of a bag-of-words style input for the dynamic topic model.
### ```dtm/analysis.py```
This module contains a class for the analysis of a fitted dynamic topic model with easy functions to access the topic representations (multinomial distribution over the vocabulary). For some examples of usage see ```../examples/dtm_analysis_examples.ipynb```.