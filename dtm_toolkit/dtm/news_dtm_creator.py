#! env/bin/python3
"""
Author: Thomas Scelsi
Email: tscelsi@student.unimelb.edu.au
File that creates a DTM-compatible dataset for the 'greyroads' dataset as mentioned in the
accompanying paper.
"""

from .creator import DTMCreator

class NewsDTMCreator(DTMCreator):
    def _extract_dates(self, date_col_name):
        return self.df[date_col_name]

def create_news_dtm_inputs(model_root, data_path, bigram=False, limit=None):
    dtm = NewsDTMCreator(model_root, data_path, text_col_name='para_text_filtered', date_col_name='year', bigram=bigram, limit=limit, already_preprocessed=True)
    dtm.preprocess_paras(write_vocab=True, ngrams=False)
    dtm.write_dtm(min_year=2010)


# if __name__ == "__main__":
    # replace
    # create_news_dtm_inputs(
    #     f"/data/news/energy-in-news-media/data/dataframes/all_aus/all_aus_filtered_min50_maxNone/dtm_model_{save_date}", 
    #     "/data/news/energy-in-news-media/data/dataframes/all_aus/all_aus_filtered_min50_maxNone/dataset.pickle",
    #     bigram=False, 
    #     limit=None,
    # )