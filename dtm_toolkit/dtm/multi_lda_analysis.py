import numpy as np
import os
from sys import exit
from numpy.lib.arraysetops import unique
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import visualisation as v
import sys
import re
# local
from dtm_toolkit.dtm.lda_analysis import Analysis

class AcrossModelAnalysis:
    """ This class analyses the similarities and differences between two LDA
    models. Here we want to achieve:
        1. Clustering of topics from both models to see which topics are similar between models. Doing
            this in a reproducible and robust way will allow us to draw conclusions of how similar topics
            from different models are impacted by changes in the other. A good example of this is seeing how
            changes in research affect changes in grey literature in similar areas.
        2. Evaluating our clustering technique by:
            a. Silhouette Analysis(?)
            b. PCA plotting to assess effectiveness of clustering and potentially between clustering methods.
        3. This clustering technique can also provide a way to evaluate our auto-assigned EuroVoc
            topics. Where topics that are clustered together should also have reasonably similar
            EuroVoc topic labels.
    
    We group the topics from each model into clusters using kmeans.
    """

    def __init__(self, m1, m2, m1_alias="M1", m2_alias="M2"):
        """
        pass two model parameter dictionaries into this class in order to create analysis classes
        for each and begin the process of analysing between the two topics. 
        """
        self.m1_alias = m1_alias
        self.m2_alias = m2_alias
        self.m1 = Analysis(
            **m1
        )
        self.m2 = Analysis(
            **m2
        )
        self.m1_common = None
        self.m2_common = None
        self._create_common_vocabulary()
        self._update_model_distributions_to_common_vocab()

    def _create_common_vocabulary(self):
        """
        In this function we create a 'common' vocabulary between the two 
        models by taking the union of the two individual model vocabularies. 
        At the same time we also create a mapping between each word in the 
        common vocabulary and it's index.
        """
        self.common_vocabulary = np.array(list(set([*self.m1.vocab, *self.m2.vocab])))
        self.w_ind_map = {w:i for i,w in enumerate(self.common_vocabulary)}
    
    def _update_model_distributions_to_common_vocab(self):
        """
        In this function, since we now have a common vocabulary between
        the two models, we want to augment their existing per-topic word 
        distributions over their vocabularies, to per-topic word distributions 
        over the shared vocabulary so that they can be compared.

        Example:

        In a simple case, we may have two models m1 and m2 that have the
        respective vocabularies:
        
        v_m1 = ["cat", "hat", "door"]
        v_m2 = ["cat", "rat", "mouse", "jerry"]

        To simplify this example, we assume that a topic for a model is represented 
        as a distribution over its vocabulary (in reality it's the sum of a topic's 
        vocabulary distribution at each time step):
        
        k_m1 = [0.7, 0.3, 0]
        k_m2 = [0.2, 0, 0.4, 0.4]

        In order to compare how similar these topics are, we create a common vocabulary 
        by computing the union over the two vocabularies of the respective models. 
        Our next step is to augment their existing distributions so that they can be 
        compared. This is done by mapping their existing distributions to the new 
        common vocabulary.

        v_common = ["cat", "hat", "door", "rat", "mouse", "jerry"]
        k_m1_common = [0.7, 0.3, 0.0001, 0, 0, 0]
        k_m2_common = [0.2, 0.001, 0, 0, 0.4, 0.4]


        we can now compute similarity measures of these same-dimension 'vectors'.
        """
        m1_common = np.zeros((self.m1.ntopics, len(self.common_vocabulary)))
        m2_common = np.zeros((self.m2.ntopics, len(self.common_vocabulary)))
        for topic_idx in range(self.m1.ntopics):
            for w, val in zip(self.m1.vocab, self.m1.betas[topic_idx]):
                m1_common[topic_idx][self.w_ind_map[w]] = val
        for topic_idx in range(self.m2.ntopics):
            for w, val in zip(self.m2.vocab, self.m2.betas[topic_idx]):
                m2_common[topic_idx][self.w_ind_map[w]] = val
        # reduce dimensionality of common vocabulary vectors
        # concat_arr = StandardScaler().fit_transform(np.concatenate((m1_common, m2_common)))
        concat_arr = np.concatenate((m1_common, m2_common))
        pca = PCA()
        dim_reduced_arr = pca.fit_transform(concat_arr)
        self.m1_common = dim_reduced_arr[:self.m1.ntopics]
        self.m2_common = dim_reduced_arr[self.m1.ntopics:self.m1.ntopics + self.m2.ntopics]
    
    def _get_similarity(self, X=None, Y=None, return_plottable=False, m1_title="Model 1 Topic", m2_title="Model 2 Topic", use_topic_labels=True):
        """
        This function computes the similarity between each topic in Model 1 against each topic in Model 2. One can either
        choose to return a plottable dataframe which can be passed into seaborn heatmap function, or the raw similarity matrix
        as return by sklearn.

        Kwargs:
            X, Y (np.ndarray, optional): If you want to pass in your own vectors from which to compute similarity, then you 
                can pass it in the X and Y variables. Both X and Y must exist for these to be valid arguments. If one of 
                these arguments is None, then the topic distributions over the common vocabularies computed 
                by instantiating this class are used.
            return_plottable (bool, optional): Flag dictating whether to return the raw sklearn similarities or a plottable dataframe. Defaults to False.
            m1_title (str, optional): This variable outlines the title to use when referencing Model 1. This will show as the axes label on the seaborn heatmap. Defaults to "Model 1 Topic".
            m2_title (str, optional): This variable outlines the title to use when referencing Model 2. This will show as the axes label on the seaborn heatmap. Defaults to "Model 2 Topic".
        """
        if type(X) == np.ndarray and type(Y) == np.ndarray:
            sim = cosine_similarity(X, Y)
        else:
            # default is the common vocabulary vectors
            sim = cosine_similarity(self.m1_common, self.m2_common)
        if return_plottable:
            t1_val = []
            t2_val = []
            sim_val = []
            if use_topic_labels:
                # get topic names for each model's topics so that they can be used in the heatmap.
                m1_topic_labels = self.m1.get_topic_labels(_type="embedding")
                m2_topic_labels = self.m2.get_topic_labels(_type="embedding")
                m1_topic_dict = {x: re.sub(r"\d{4} ", "", y[0][0])+str(x) for x, y in enumerate(m1_topic_labels)}
                m2_topic_dict = {x: re.sub(r"\d{4} ", "", y[0][0])+str(x) for x, y in enumerate(m2_topic_labels)}
            for t1_ind, matrix in enumerate(sim):
                for t2_ind, val in enumerate(matrix):
                    t1_val.append(t1_ind)
                    t2_val.append(t2_ind)
                    sim_val.append(val)
            m1_title = self.m1_alias if self.m1_alias else m1_title
            m2_title = self.m2_alias if self.m2_alias else m2_title
            df = pd.DataFrame({m1_title: t1_val, m2_title: t2_val, "Similarity": sim_val})
            if use_topic_labels:
                df[m1_title] = df[m1_title].map(m1_topic_dict)
                df[m2_title] = df[m2_title].map(m2_topic_dict)
            self.heatmap_data = df.pivot(index=m1_title, columns=m2_title, values='Similarity')
            return self.heatmap_data
        else:
            return sim
    
    def get_similar_topics(self, threshold=0.5, fp=sys.stdout, **kwargs):
        """This function computes the cosine similarity between topics from two models. If the similarity
        is greater than the threshold argument between two topics from the different models, 
        then they are saved to a file along with the top words for each topic. This is a handy function to quickly
        find topics between models that you should compare further.

        Args:
            threshold (float, optional): Similarity threshold where if greater, two topics are considered similar. Defaults to 0.5.
            fp ([type], optional): The open file handler to write the similar topic information to. Defaults to sys.stdout.
        
        Kwargs:
            All kwargs go into the DTMAnalysis.get_top_words function, see the appropriate documentation for that function in analysis.py
        """
        fp.write("#"*10 + "\n")
        fp.write(f"## FINDING SIMILAR TOPICS BETWEEN {self.m1_alias} AND {self.m2_alias} ##\n")
        fp.write("#"*10 + "\n")
        res = self._get_similarity(return_plottable=False)
        self.m1.get_top_words(with_prob=False, **kwargs)
        self.m2.get_top_words(with_prob=False, **kwargs)
        for m1_topic_ind in range(len(res)):
            for m2_topic_ind in range(len(res[m1_topic_ind])):
                if res[m1_topic_ind][m2_topic_ind] > threshold:
                    fp.write(f"{self.m1_alias} topic {m1_topic_ind} and {self.m2_alias} topic {m2_topic_ind} are similar (sim={res[m1_topic_ind][m2_topic_ind]}).\n")
                    fp.write(f"{self.m1_alias} topic {m1_topic_ind} top words:\n")
                    fp.write(str(self.m1.top_word_arr[m1_topic_ind])+"\n")
                    fp.write(f"{self.m2_alias} topic {m2_topic_ind} top words:\n")
                    fp.write(str(self.m2.top_word_arr[m2_topic_ind])+"\n")
                    fp.write("==========\n")

    def get_unique_topics(self, threshold=0.25, fp=sys.stdout, **kwargs):
        """This function return the topics from both model 1 that are not similar to any 
        topics within model 2 and vice-versa. Thus allowing us to see topics that are 
        unique between publications/datasets. i.e. if a topic appears in model 1 and not 
        in model 2, then that topic is unique to the discussions found in model 1's dataset.

        Args:
            threshold (float, optional): Similarity threshold where if no topic from the opposite model is above this threshold, 
                a topic is considered unique. Defaults to 0.5.
            fp ([type], optional): The open file handler to write the unique topic information to. Defaults to sys.stdout.

        """
        fp.write("#"*10 + "\n")
        fp.write(f"## FINDING UNIQUE TOPICS IN {self.m1_alias} AND {self.m2_alias} ##\n")
        fp.write("#"*10+ "\n")
        def print_unique_topic(ind, max_sim, model_name):
            if model_name == "m1":
                model = self.m1
                alias = self.m1_alias
            else:
                model = self.m2
                alias = self.m2_alias
            fp.write(f"Topic {ind} in {alias} is unique (max_sim={max_sim}).\n")
            fp.write("---\n")
            fp.write(f"Topic label: {model.topic_labels[ind]}\n")
            fp.write("---\n")
            fp.write(f"Topic word list: {model.top_word_arr[ind]}\n")
            fp.write("==========\n")
        m1_unique_topics = []
        m2_unique_topics = []
        res = self._get_similarity(return_plottable=False)
        res_T = res.T
        self.m1.get_topic_labels(**kwargs)
        self.m2.get_topic_labels(**kwargs)
        for m1_topic_ind in range(len(res)):
            max_sim = res[m1_topic_ind].max()
            if max_sim <= threshold:
                m1_unique_topics.append(m1_topic_ind)
                print_unique_topic(m1_topic_ind, max_sim, "m1")
        for m2_topic_ind in range(len(res_T)):
            max_sim = res_T[m2_topic_ind].max()
            if max_sim <= threshold:
                m2_unique_topics.append(m2_topic_ind)
                print_unique_topic(m2_topic_ind, max_sim, "m2")

    # def _compare_with_gold_standard(self, X, gold):
    #     total_sim = self._get_similarity(X, gold, return_plottable=False)
    #     total_sim_T = self._get_similarity(X.T, gold.T, return_plottable=False)
    #     comparison_vals = []
    #     # assuming that both models have same number of topics...
    #     for i in range(len(total_sim)):
    #         comparison_vals.append(total_sim[i][i])
    #         comparison_vals.append(total_sim_T[i][i])

    #     average_similarity_across_methods = np.array(comparison_vals).mean()
    #     # print("==========")
    #     # print(f"{name} has mean similarity to gold standard of: {average_similarity_across_methods}")
    #     return average_similarity_across_methods
    
    def get_heatmap(self, **kwargs):
        """This function computes the similarity between the two models across all topics and
        visualises that as a heatmap.

        Returns:
            [type]: [description]
        """
        save_path = kwargs.pop("save_path", None)
        res = self._get_similarity(return_plottable=True, **kwargs)
        v.heatmap(res, save_path)
        return res

def compare_all_models():
    models = [{
        "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "dtm", "final_datasets", "greyroads_ieo_min_freq_40_1997_2020_ngram"),
        "model_out_dir": "k30_a0.01_var0.05", 
        "doc_year_map_file_name": "model-year.dat",
        "seq_dat_file_name": "model-seq.dat",
        "vocab_file_name": "vocab.txt",
        "eurovoc_whitelist": True,
        "alias": "IEO"
    }, {
        "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "dtm", "final_datasets", "greyroads_aeo_min_freq_40_1997_2020_ngram"),
        "model_out_dir": "k30_a0.01_var0.05", 
        "doc_year_map_file_name": "model-year.dat",
        "seq_dat_file_name": "model-seq.dat",
        "vocab_file_name": "vocab.txt",
        "eurovoc_whitelist": True,
        "alias": "AEO"
    }, 
    # {
    #     "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "dtm", "datasets", "journal_energy_policy_applied_energy_all_years_abstract_all_ngram"),
    #     "model_out_dir": "k30_a0.01_var0.05", 
    #     "doc_year_map_file_name": "model-year.dat",
    #     "seq_dat_file_name": "model-seq.dat",
    #     "vocab_file_name": "vocab.txt",
    #     "eurovoc_whitelist": True,
    #     "alias": "Journals - All"
    # }, {
    #     "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "dtm", "datasets", "journal_energy_policy_applied_energy_all_years_abstract_biofuels_ngram_min_freq_50"),
    #     "model_out_dir": "k30_a0.01_var0.05", 
    #     "doc_year_map_file_name": "model-year.dat",
    #     "seq_dat_file_name": "model-seq.dat",
    #     "vocab_file_name": "vocab.txt",
    #     "eurovoc_whitelist": True,
    #     "alias": "Journals - Biofuels"
    # }, {
    #     "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "dtm", "datasets", "journal_energy_policy_applied_energy_all_years_abstract_solar_ngram_min_freq_50"),
    #     "model_out_dir": "k30_a0.01_var0.05", 
    #     "doc_year_map_file_name": "model-year.dat",
    #     "seq_dat_file_name": "model-seq.dat",
    #     "vocab_file_name": "vocab.txt",
    #     "eurovoc_whitelist": True,
    #     "alias": "Journals - Solar"
    # }, {
    #     "model_root": os.path.join(os.environ['ROADMAP_SCRAPER'], "DTM", "dtm", "datasets", "journal_energy_policy_applied_energy_all_years_abstract_coal_ngram_min_freq_50"),
    #     "model_out_dir": "k30_a0.01_var0.05", 
    #     "doc_year_map_file_name": "model-year.dat",
    #     "seq_dat_file_name": "model-seq.dat",
    #     "vocab_file_name": "vocab.txt",
    #     "eurovoc_whitelist": True,
    #     "alias": "Journals - Coal"
    # }
    ]
    m1_list = []
    m2_list = []
    # simple_base_res, thresh_base_res, top_ten_base_res, thresh_zeroed_base_res, top_ten_zeroed_base_res, simple_ev_res, tfidf_ev_res
    for i in range(len(models)):
        for j in range(len(models)):
            if j <= i:
                continue
            m1 = models[i]
            m2 = models[j]
            m1_name = m1.pop('alias')
            m2_name = m2.pop('alias')
            print(f"{m1_name} vs {m2_name}")
            m1_list.append(m1_name)
            m2_list.append(m2_name)
            ama = AcrossModelAnalysis(m1, m2, m1_alias=m1_name, m2_alias=m2_name)
            res = get_summary(ama)
            
    
def get_summary(ama, save_dir):
    similar_topics_path = os.path.join(save_dir, "similar_topics")
    unique_topics_path = os.path.join(save_dir, "unique_topics")
    if not os.path.isdir(similar_topics_path):
        os.mkdir(similar_topics_path)
    if not os.path.isdir(unique_topics_path):
        os.mkdir(unique_topics_path)
    with open(os.path.join(similar_topics_path, f"{ama.m1_alias}_{ama.m2_alias}_similar_topics.txt"), "w+") as fp:
        ama.get_similar_topics(fp=fp, threshold=0.75)
    with open(os.path.join(unique_topics_path, f"{ama.m1_alias}_{ama.m2_alias}_unique_topics.txt"), "w+") as fp:
        ama.get_unique_topics(fp=fp)
    ama.get_heatmap(save_path=f"{os.path.join(save_dir, '_'.join(ama.m1_alias.split(' ')) + '_v_' + '_'.join(ama.m2_alias.split(' ')))}")
    # return res

if __name__ == "__main__":
    m1 = {
        "corpus_path": "/data/news/energy-in-news-media/data/dataframes/nch/nch_filtered_min150_maxNull/dataset.pickle", 
        "model_root": "/data/news/energy-in-news-media/data/dataframes/nch/nch_filtered_min150_maxNull/model_04012022_11:15:43",
        "alias": "min0_maxNone"
    }
    m2 = {
        "corpus_path": "/data/news/energy-in-news-media/data/dataframes/nch/nch_filtered_min150_max10000/dataset.pickle", 
        "model_root": "/data/news/energy-in-news-media/data/dataframes/nch/nch_filtered_min150_max10000/model_04012022_11:13:59",
        "alias": "min150_max10000"
    }
    ama = AcrossModelAnalysis(m1, m2, m1.pop('alias'), m2.pop('alias'))
    get_summary(ama, "/data/news/energy-in-news-media/data/models/ctm/ama_analysis")