#! env/bin/python3
"""
Author: Thomas Scelsi
Email: tscelsi@student.unimelb.edu.au
File that creates a DTM-compatible dataset for the 'greyroads' dataset as mentioned in the
accompanying paper.
"""

from .creator import DTMCreator
import pandas as pd
from datetime import timedelta

class NewsDTMCreator(DTMCreator):
    def _extract_dates(self, date_col_name):
        return self.df[date_col_name].apply(lambda x: f"{x.year}{x.month:02}")

def create_year_mapping(series: pd.Series):
    year_month_list = series.sort_values().apply(lambda x: f"{x.year}{x.month:02}").drop_duplicates().tolist()
    mapping = {k:v//3 for v,k in enumerate(year_month_list)}
    return mapping

def create_news_dtm_inputs(model_root, data_path, limit=None):
    dtm = NewsDTMCreator(model_root, data_path, text_col_name='para_text_filtered', date_col_name='datepublished', limit=limit)
    year_mapping = create_year_mapping(dtm.df.datepublished)
    dtm.preprocess_paras(write_vocab=True, min_freq=0, min_num_toks_per_doc=0, min_unique_toks_per_doc=0, timestep_mapping=year_mapping)
    dtm.write_dtm()


# if __name__ == "__main__":
    # replace
    # create_news_dtm_inputs(
    #     f"/data/news/energy-in-news-media/data/dataframes/all_aus/all_aus_filtered_min50_maxNone/dtm_model_{save_date}", 
    #     "/data/news/energy-in-news-media/data/dataframes/all_aus/all_aus_filtered_min50_maxNone/dataset.pickle",
    #     bigram=False, 
    #     limit=None,
    # )