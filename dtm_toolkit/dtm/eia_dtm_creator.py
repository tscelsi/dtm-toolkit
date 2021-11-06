#! env/bin/python3
"""
Author: Thomas Scelsi
Email: tscelsi@student.unimelb.edu.au
File that creates a DTM-compatible dataset for the 'greyroads' dataset as mentioned in the
accompanying paper.
"""

from .creator import DTMCreator
import os

class EIADTMCreator(DTMCreator):
    def _extract_dates(self, date_col_name):
        return [int(x) for x in self.df[date_col_name].tolist()]

def create_greyroads_inputs(model_root, csv_path, bigram=True, limit=None, min_freq=20, ngrams=True):
    jdtmc = EIADTMCreator(model_root, csv_path, text_col_name='para_text', date_col_name='year', bigram=bigram, limit=limit)
    jdtmc.preprocess_paras(min_freq=min_freq, write_vocab=True, ngrams=ngrams, save_preproc=True)
    jdtmc.write_dtm()


if __name__ == "__main__":
    # replace
    create_greyroads_inputs(
        "/data/greyroads/energy-roadmap/DTM/dtm/toolkit_test", 
        os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "greyroads_ieo_1997_2020.csv"), 
        bigram=True, 
        limit=None,
        min_freq=40
    )