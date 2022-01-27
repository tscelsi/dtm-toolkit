"""
    -- analysis.py -- 
    
    This file contains logic and functionality for analysing
    the results of a model created by the Dynamic Topic Model (DTM). We use this
    model to understand what sorts of topics appear in a corpus of text, and
    also to understand how the prevalence of these topics fluctuates over time. 
"""

import pandas as pd
import os
import numpy as np
import spacy
from .eurovoc import Eurovoc
from ..auto_labelling import AutoLabel


class Analysis:

    def __init__(
        self,  
        corpus_path,
        model_root,
        theta_file_name="thetas.npy",
        vocab_file_name="vocab.txt",
        beta_file_name="top_word_dist.npy",
        model_out_dir="analysis_out",
        thesaurus=None,
        thes_phrase_col="TERMS (PT-NPT)",
        thes_label_col="MT",
        spacy_lang="en_core_web_sm"
        ):
        self.nlp = spacy.load(spacy_lang)
        try:
            self.corpus = pd.read_pickle(corpus_path)
        except:
            self.corpus = pd.read_csv(corpus_path)
        # number of paragraphs in this corpus
        self.ndocs = len(self.corpus)
        # load thesaurus for auto-labelling
        if thesaurus:
            self.thesaurus = thesaurus
        else:
            # defaults to Eurovoc as per original paper
            self.thesaurus = Eurovoc(eurovoc_whitelist=True).eurovoc
        # load auto-labeling module
        self.auto_labelling = AutoLabel(self.thesaurus, phrase_col=thes_phrase_col, label_col=thes_label_col, spacy_lang=spacy_lang)
        
        self.model_root = model_root
        self.model_out_dir = model_out_dir
        
        # load model results
        # thetas represent the document-topic mixtures
        self.thetas = np.load(os.path.join(model_root, theta_file_name))
        # betas represent the per-topic prob distribution over vocabulary
        self.betas = np.load(os.path.join(model_root, beta_file_name))
        self.ntopics = self.betas.shape[0]
        
        # load vocab
        vocab_file = os.path.join(self.model_root, vocab_file_name)
        self.vocab = np.array([x.strip() for x in open(vocab_file).readlines()])
        
        self.index_to_word = {i:w for i, w in enumerate(self.vocab)}


    # def save_gammas(self, save_path, split=True):
    #     doc_ids = pd.read_csv(os.path.join(self.model_root, "doc_ids.csv"))
    #     assert len(doc_ids) == len(self.doc_topic_gammas)
    #     if split:
    #         tmp_df = pd.DataFrame(self.doc_topic_gammas['topic_dist'].tolist(), columns=[i for i in range(self.ntopics)])
    #         tmp_df['year'] = self.doc_topic_gammas['year']
    #         tmp_df['doc_id'] = doc_ids['doc_id']
    #         tmp_df.to_csv(save_path)
    #         del tmp_df
    #     else:
    #         self.doc_topic_gammas['doc_id'] = doc_ids['doc_id']
    #         self.doc_topic_gammas.to_csv(save_path)
    
    def get_doc_topic_mixtures(self):
        return self.doc_topic_gammas

    def create_plottable_topic_proportion_ot_df(self, remove_small_topics=False, threshold=0.01, merge_topics=False, include_names=False, limit=2020, **kwargs):
        """[summary]

        Args:
            remove_small_topics (bool, optional): [description]. Defaults to False.
            threshold (float, optional): [description]. Defaults to 0.01.
            merge_topics (bool, optional): [description]. Defaults to False.
            include_names (bool, optional): [description]. Defaults to False.
            limit (int, optional): [description]. Defaults to 2020.

        Returns:
            [type]: [description]
        """
        df = self._create_topic_proportions_per_year_df(remove_small_topics, threshold, merge_topics=merge_topics, include_names=include_names, **kwargs)
        df = df[df['year'] <= limit]
        df = df.pivot(index='year', columns='topic_name', values='proportion')
        return df
    
    def get_topic_label_proportions_ot(self, topic_idx, return_type="matrix", normalised=True, n=10, **kwargs):
        word_dist_arr_ot = self.get_topic_representation_ot(topic_idx)
        top_words_ot = self._get_words_for_topic(word_dist_arr_ot, over_time=True, n=n)
        label_names = []
        for tw in top_words_ot:
            topic_scores = self.auto_labelling._get_auto_topic_name(tw, topic_idx, score_type="embedding", raw=True, **kwargs)
            total_score = sum(topic_scores.values())
            if return_type == "matrix":
                    if normalised:
                        label_names.append(np.array([topic_scores[label]/total_score for label in self.auto_labelling.sorted_labels]))
                    else:
                        label_names.append(np.array([topic_scores[label] for label in self.auto_labelling.sorted_labels]))
            else:
                if normalised:
                    label_names.append(dict([(l, score / total_score) for l,score in topic_scores.items()]))
                else:
                    label_names.append(topic_scores)
        if return_type == "matrix":
            return np.array(label_names)                
        return label_names

    def get_topic_labels(self, _type="tfidf", raw=False, n=4, **kwargs):
        """
        This is the main function used to retrieve automatic labels for each DTM topic.
        """
        top_words = self.get_top_words(with_prob=True)
        topic_labels = self.auto_labelling.get_topic_labels(top_words, top_n=n, score_type=_type, raw=raw, **kwargs)
        self.topic_labels = topic_labels
        return topic_labels
    
    def get_top_words(self, n=10, with_prob=False):
        """
        Gets the top words for each topic within the CTM model being analysed.
        """
        self.top_word_arr = []
        for i, topic_probs in enumerate(self.betas):
            inds = topic_probs.argsort()[::-1][:n]
            if with_prob:
                self.top_word_arr.append([(x,y) for x,y in zip(topic_probs[inds], self.vocab[inds])])
            else:
                self.top_word_arr.append([x for x in self.vocab[inds]])
        return self.top_word_arr

    def get_top_words_ot(self, topic_idx=None, n=10):
        pass