#! env/bin/python3
"""
Author: Thomas Scelsi
Email: tscelsi@student.unimelb.edu.au
File that creates a DTM-compatible dataset for the 'greyroads' dataset as mentioned in the
accompanying paper.
"""

from .creator import DTMCreator

class TwitterDTMCreator(DTMCreator):
    def _extract_dates(self, date_col_name):
        return self.df[date_col_name]

def create_twitter_dtm_inputs(model_root, data_path, min_freq, iterator=None, bigram=False, limit=None):
    dtm = TwitterDTMCreator(model_root, data_path, text_col_name='preproc_sentence', date_col_name='unit', bigram=bigram, limit=limit, already_preprocessed=True)
    dtm.preprocess_paras(write_vocab=True, ngrams=False, min_freq=0)
    dtm.write_dtm(year_iterator=iterator)


# if __name__ == "__main__":
    # replace
    # create_twitter_dtm_inputs(
    #                 os.path.join(save_dir, f"dtm_model_{save_date}"),
    #                 os.path.join(save_dir, "dataset.pickle"),
    #                 iterator=['202004', '202005', '202006', '202007', '202008', '202009', '202010', '202011', '202012', '202101', '202102', '202103', '202104', '202105', '202106', '202107', '202108', '202109', '202110', '202111', '202112', '202201'],
    #                 bigram=False,
    #                 limit=None,
    # )