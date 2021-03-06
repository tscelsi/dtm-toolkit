#! env/bin/python3
# Thomas Scelsi
# Full set of dynamic topic modelling tools for a general dataset

import pandas as pd
import os
import spacy
import re
from typing import List, Optional, Iterable, Union
from numpy import random
from collections import defaultdict, Counter
import json
from multiprocessing import Pool
from dtm_toolkit.preprocessing import Preprocessing
from pprint import pprint
import sys
from dotenv import load_dotenv
load_dotenv()

SEED = 42

class DTMCreator:
    
    term_blacklist = []
    
    def __init__(
        self, 
        model_root: str, 
        docs: str,
        text_col_name: Optional[str]='section_txt', 
        date_col_name: Optional[str]='date',
        doc_id_col_name: Optional[str]='doc_id',
        limit=None, 
        years_per_step=1,
        shuffle=True,
        spacy_batch_size=256,
    ):
        self.spacy_batch_size = spacy_batch_size
        if isinstance(docs, str):
            # this is assumed to be path to df
            if docs.endswith(".tsv"):
                self.df = pd.read_csv(docs, sep="\t")
            elif docs.endswith(".pickle"):
                self.df = pd.read_pickle(docs)
            else:
                # default
                self.df = pd.read_csv(docs)
            self.doc_id_col_name = doc_id_col_name
            self.text_col_name = text_col_name
            self.df = self.df.dropna(subset=[text_col_name])
            self.years = self._extract_dates(date_col_name)
        else:
            print("Need to pass a path to csv containing corpus information. See examples for more details.")
            sys.exit(1)
        self.df['year'] = self.years
        self.years_per_step = years_per_step
        # create directory structure
        if not os.path.isdir(model_root) and model_root != "":
            os.mkdir(model_root)
        self.model_root = model_root
        # spacy load
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.remove_pipe("parser")
        self.nlp.remove_pipe("ner")
        self.nlp.add_pipe('sentencizer')
        self.rdocs =[]
        self.rdates = []
        rand_indexes = [idx for idx in random.RandomState(SEED).permutation(len(self.df))] if shuffle else range(len(self.df))
        if limit:
            self.df = self.df.iloc[rand_indexes[:limit]]
        else:
            self.df = self.df.iloc[rand_indexes]
        self.rdocs = self.nlp.pipe(self.df[text_col_name], n_process=11, batch_size=self.spacy_batch_size)
        return

    @classmethod
    def get_paras_from_mult_dat(self, mult_path, vocab_path):
        vocab = [x.split("\t")[0] for x in open(vocab_path, "r").readlines()]
        paras = []
        for line in open(mult_path, "r").readlines():
            para = []
            for index, count in [x.split(":") for x in line.split(" ")[1:]]:
                para.append([vocab[int(index)].strip() for _ in range(int(count))])
            paras.append(para)
        return paras

    def _extract_dates(self, date_col_name):
        """Here we extract the year for each document by taking a particular column of the dataframe containing the corpus
        and extracting the year from a date stamp. This will need to be overidden depending on the format of your date and the column name. 
        In this case, the df has a column labelled 'date' which contains a date of form 2020-10-09. 
        We extract the year (2020) for each doc in the df.

        A simple example function could be:
            return [int(d.split("-")[0]) for d in self.df[date_col_name].tolist()]

            when dates are of the form: 2020-01-01
        """
        raise NotImplementedError

    def _get_year_batches(self, years_list=None):
        years = years_list if years_list else self.df['year']
        year_mapping = {}
        for year in years:
            batch_num = int((year - min(years)) / self.years_per_step)
            year_mapping[year] = batch_num
        return year_mapping

    def batch_years(self, years_list=None, return_mapping=False):
        """
        This function takes an already created DTM input sequence of years
        and gives back the same years batched into spans of years based on the _get_year_batches
        function.
        """
        years = years_list if years_list else self.rdates
        year_mapping = self._get_year_batches(years)
        new_years_list = [year_mapping[year] for year in years]
        if return_mapping:
            return new_years_list, year_mapping
        else:
            return new_years_list

    def preprocess_paras(
            self,
            min_freq=150,
            min_num_toks_per_doc=15,
            min_unique_toks_per_doc=5,
            timestep_mapping={},
            write_vocab=False, 
        ):
        """This function takes the spaCy documents found in this classes rdocs attribute and preprocesses them.
        The preprocessing pipeline tokenises each document and removes:
        1. punctuation
        2. spaces
        3. numbers
        4. urls
        5. stop words and single character words.

        It then lemmatises and lowercases each token and joins multi-word tokens together with an _. i.e. 'climate change' becomes 'climate_change' in case this was not
        done in the _add_bigrams function.

        Once the tokens have been created, token frequency counts are taken and a vocabulary of tokens and their counts are created.
        Ensuring that tokens only above a certain min_freq are kept in the vocabulary.

        Finally class attributes are created in preparation for being input into a DTM. This requires a mult.dat and seq.dat file (see DTM documentation).

        To save these class attributes you need to run self.write_dtm

        Args:
            min_freq (int, optional): Minimum count frequency of tokens in vocabulary. Defaults to 150.
            write_vocab (bool, optional): Whether or not to write vocabulary to file. Defaults to False.
            ds_lower_limit (int, optional): If downsampling is enabled, this is the number that a particular timesteps documents will be downsampled to. Defaults to 1000.
            us_upper_limit (int, optional): If upsampling is enabled, the documents for a particular timestep will be increased to this number. Defaults to 200.
            enable_downsampling (bool, optional): Whether or not downsampling is enabled. Defaults to False.
            enable_upsampling (bool, optional): Whether or not upsampling is enabled. Defaults to False.
            ngrams (bool, optional): Whether to add ngrams instead of just bigrams. Relies on self.bigrams to be True to have any effect. Defaults to False.
            basic (bool, optional): Whether to just undertake the basic preprocessing step to create the paras_processed list only. Defaults to False.
        """
        self.paras_processed = []
        self.doc_index_order = []
        self.timestep_mapping = timestep_mapping
        wids = {}
        wids_rev = {}
        self.wcounts = defaultdict(lambda:0)
        self.paras_processed = self.df[self.text_col_name]
        if isinstance(self.paras_processed.iloc[0], str):
            self.paras_processed = [x.split(" ") for x in self.paras_processed]
        assert isinstance(self.paras_processed, list)
        # count words
        for d in self.paras_processed:
            for w in d:
                self.wcounts[w]+=1
        # PREPROCESS: keep tokens that occur at least min_freq times
        self.wcounts = {k:v for k,v in self.wcounts.items() if v>min_freq} 
        # collect word IDs
        for d in self.paras_processed:
            for w in d:
                if w in self.wcounts and w not in wids:
                    wids_rev[len(wids)]=w
                    wids[w]=len(wids)
        if write_vocab:
            with open(os.path.join(self.model_root, "vocab.txt"), 'w+') as of:
                for i in range(len(wids_rev)):
                    assert wids[wids_rev[i]]==i
                    of.write(f"{wids_rev[i]}\t{self.wcounts[wids_rev[i]]}\n")
        # transform to DTM input
        self.paras_to_wordcounts = []
        self.timesteps_final = []
        # if we need to merge years, then it is done through the years_per_step var
        final_df_mask = []
        for idx, doc in enumerate(self.paras_processed):
            token = [w for w in doc if w in self.wcounts]
            type_counts = Counter(token)
            # PREPROCESS: at least 15 token and >5 types per document
            if len(token) > min_num_toks_per_doc and len(type_counts) > min_unique_toks_per_doc:
                id_counts = [f"{len(type_counts)}"]+[f"{wids[k]}:{v}" for k,v in type_counts.most_common()]
                self.paras_to_wordcounts.append(' '.join(id_counts))
                final_df_mask.append(True)
                if self.timestep_mapping != {}:
                    self.timesteps_final.append(self.timestep_mapping[self.df['year'].iloc[idx]])
                else:
                    self.timesteps_final.append(self.df['year'].iloc[idx])
            else:
                final_df_mask.append(False)
        self.df = self.df[final_df_mask]


    def write_dtm(self, min_year=None, max_year=None, year_iterator: Optional[Iterable[Union[str, int]]]=None, write_csv=False):
        """Write the mult.dat and seq.dat and year.dat files needed to fit the DTM

        Args:
            min_year (int, optional): If years_per_step is 1, can specify a particular cutoff min year to write to files. Defaults to min(self.dates).
            max_year (int, optional): If years_per_step is 1, can specify a particular cutoff max year to write to files. Defaults to max(self.dates).
        """
        # write -mult file and -seq file
        outmult = open(os.path.join(self.model_root, "model-mult.dat"), 'w+')
        outyear = open(os.path.join(self.model_root, "model-year.dat"), 'w+')
        outseq = open(os.path.join(self.model_root, "model-seq.dat"), 'w+')

        ordered_doc_ids = []
        print(len(self.timesteps_final))
        print(len(self.paras_to_wordcounts))

        stepcount = defaultdict(lambda:0)

        if self.timestep_mapping == {}:
            min_date = min_year if min_year else min(self.df['year'])
            max_date = max_year if max_year else max(self.df['year'])
            timestep_iterator = range(min_date, max_date + 1, 1)
        else:
            with open(os.path.join(self.model_root,"timestep_mapping.json"), "w") as fp:
                json.dump(self.timestep_mapping, fp)
            timestep_iterator = sorted(list(set(self.timestep_mapping.values())))
        for step in timestep_iterator:
            for idx, yy in enumerate(self.timesteps_final):
                if step == yy:
                    stepcount[step] += 1
                    outyear.write(f"{str(yy)}\n")
                    outmult.write(f"{self.paras_to_wordcounts[idx]}\n")
                    ordered_doc_ids.append(idx)

        outseq.write(f"{len(stepcount)}\n")

        timestep_dict = {}
        for step in sorted(timestep_iterator):
            outseq.write(f"{stepcount[step]}\n")
            timestep_dict[len(timestep_dict)] = step
        self.df = self.df.iloc[ordered_doc_ids]
        self.df.to_pickle(os.path.join(self.model_root, "ordered_dtm.pickle"))

        outyear.close()
        outmult.close()
        outseq.close()
        # outdocids.close()
        


def fit(run, model_root_dir, outpath):
    print(outpath)
    if not os.path.isdir(outpath):
        print("making dir")
        os.mkdir(outpath)
    cmd = os.path.join(os.environ['DTM_ROOT'], "dtm", "dtm", "main") + f" --ntopics={run['topics']}   --mode=fit   --rng_seed=0   --initialize_lda=true   --corpus_prefix={os.path.join(model_root_dir, 'model')}   --outname={outpath}   --top_chain_var={run['topic_var']}   --alpha={run['alpha']}   --lda_sequence_min_iter=6   --lda_sequence_max_iter=20   --lda_max_em_iter=10"
    print(cmd)
    os.system(cmd)
    return 1

def fit_mult_model(model_root_dir):
    with open(os.path.join(os.environ['DTM_ROOT'], "runs.json"), "r") as fp:
        data = json.load(fp)
    with Pool(processes=12) as pool:
        multiple_results = [pool.apply_async(fit, (run, model_root_dir, os.path.join(model_root_dir, f"k{run['topics']}_a{run['alpha']}_var{run['topic_var']}"))) for run in data]
        for res in multiple_results:
            print(res.get())
    return 1

def fit_one():
    run = {"alpha": 0.01,
        "topic_var": 0.05,
    "topics": 30}
    model_root_dir = os.path.join(os.environ['DTM_ROOT'], "dtm", "dataset_2a")
    outpath = os.path.join(model_root_dir, f"model_run_topics{run['topics']}_alpha{run['alpha']}_topic_var{run['topic_var']}")
    fit(run, model_root_dir=model_root_dir, outpath=outpath)

if __name__ == "__main__":
    # the path to the directory that you want all the files saved in, e.g. *-mult.dat, *-seq.dat, vocab.txt, etc.
    # model_root = os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "journal_energy_policy_applied_energy_all_years_abstract_biofuels_bigram")
    # # the path to the journal paragraphs that are to become part of the fitting data for the topic model.
    # data_path = os.path.join(os.environ['ROADMAP_SCRAPER'], "journals", "journals_energy_policy_applied_energy_all_years_abstract_biofuels.csv")
    # create_model_inputs("tom_test", os.path.join(os.environ['HANSARD'], "coal_data", "04_model_inputs", "coal_full_downloaded.csv"), text_col_name="main_text", date_col_name="date", bigram=False, limit=100)
    # create_mult_datasets()
    # fit_mult_datasets()
    fit_mult_model(os.path.join(os.environ['DTM_ROOT'], "dtm", "datasets", "all_2a_last_20_years_min_freq_80_21_09_21"))
    # fit_one()