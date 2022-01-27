import os
from typing import Optional
import pandas as pd
from dtm_toolkit.dtm.analysis import DTMAnalysis
from dtm_toolkit.dtm.eurovoc import Eurovoc
from pprint import pprint
import re
import numpy as np

def analyse(analysis: DTMAnalysis, model_df: pd.DataFrame, model_name: str, save_dir: str, plot: Optional[bool]=True, coherence: Optional[bool]=True, terms: Optional[bool]=True, document_topic_dist: Optional[bool]=True):
    analysis_save_dir = save_dir
    # topic_analysis_save_dir = os.path.join(analysis_save_dir, "topic_analysis")
    model_topic_analysis_save_dir = os.path.join(analysis_save_dir, model_name)
    if not os.path.isdir(analysis_save_dir):
        os.mkdir(analysis_save_dir)
    if not os.path.isdir(model_topic_analysis_save_dir):
        os.mkdir(model_topic_analysis_save_dir)
    if document_topic_dist:
        analysis.save_gammas(model_df._id, os.path.join(model_topic_analysis_save_dir, "doc_topic_distribution.csv"))
    if plot:
    # plot the topic distributions over time for this model
        print("plotting topics...")
        try:
            # coh.plot_topics_ot(os.path.join(model_topic_analysis_save_dir, f"{model}.png"), scale=1.2)
            analysis.plot_topics_ot(os.path.join(model_topic_analysis_save_dir, f"{model_name}_merged.png"), merge_topics=True, scale=1.2, _type="embedding", sort_by=sort_cols, fontsize=50)
        except Exception as e:
            print(e)
            print(f"plot failed for model: {model_name}")
            breakpoint()
    # if coherence:
    #     # get the coherence of this model
    #     print("calculating coherence...")
    #     pmi_coh = analysis.get_coherence()
    #     npmi_coh = analysis.get_coherence("c_npmi")
    #     coherences = {}
    #     coherences[model] = {}
    #     coherences[model]['pmi'] = pmi_coh
    #     coherences[model]['npmi'] = npmi_coh
    #     with open(os.path.join(analysis_save_dir, "coherences.txt"), "w") as fp:
    #         fp.write("Model\tPMI\tNPMI\n")
    #         for k,v in coherences.items():
    #             fp.write(f"{k}\t{v['pmi']}\t{v['npmi']}\n")
    if terms:
        tfidf_topic_names = analysis.get_topic_labels()
        emb_topic_names = analysis.get_topic_labels(_type="embedding")
        topic_proportions_per_year = pd.DataFrame(analysis._get_topic_proportions_per_year().tolist(), index=analysis._get_topic_proportions_per_year().index)
        topw_df = analysis.get_top_words_ot(n=30)
        with open(os.path.join(model_topic_analysis_save_dir, "all_topics_top_terms.txt"), "w") as fp1, \
            open(os.path.join(model_topic_analysis_save_dir, f"all_topics_top_terms_ot.txt"), "w") as fp2:
            for i in range(len(tfidf_topic_names)):
                topic_top_terms = analysis.top_word_arr[i]
                tfidf_topic_name = tfidf_topic_names[i]
                emb_topic_name = emb_topic_names[i]
                # word_dist_arr_ot = coh.get_topic_word_distributions_ot(i)
                # topic_top_terms = coh.get_words_for_topic(word_dist_arr_ot, with_prob=False)
                fp1.write(f"tfidf topic {i} labels: ({tfidf_topic_name})\nemb topic {i} labels: ({emb_topic_name})\n{topic_top_terms}\n==========\n")
                fp2.write(f"\n=========\ntfidf topic {i} labels: ({tfidf_topic_name})\nemb topic {i} labels: ({emb_topic_name})\n=========\n\n")
                top_words_for_topic = topw_df[topw_df['topic_idx'] == i].loc[:, ['year', 'top_words']]
                top_words_for_topic['proportion'] = topic_proportions_per_year[i].tolist()
                for row in top_words_for_topic.itertuples():
                    proportion = '{:05.3f}'.format(row.proportion)
                    fp2.write(f"{row.year}\t{proportion}\t{row.top_words}\n")
                
def analyse_multi_models():

    # Load thesaurus here if you want, else use default by not passing anything to DTMAnalysis class.
    whitelist_ev_labels = [x.strip() for x in open("/data/news/energy-in-news-media/hansard_eurovoc_subdomain_labels.txt").readlines()]
    thesaurus = Eurovoc(eurovoc_whitelist=True, whitelist_eurovoc_labels=whitelist_ev_labels, remap=False).eurovoc

    datasets = [
        {
            "model_root": "/data/news/energy-in-news-media/data/dataframes/all_aus/all_aus_filtered_min50_max20000_minyear_2012/dtm_model_18012022_03:55:56",
            "data_path": "/data/news/energy-in-news-media/data/dataframes/all_aus/all_aus_filtered_min50_max20000_minyear_2012/dtm_model_18012022_03:55:56/ordered_dtm.pickle",
        },
    ]
    whitelist_k = [30]
    whitelist_a = [0.01]
    whitelist_var = [0.1, 0.05, 0.01, 0.001]
    for ds in datasets:
        dataset_df = pd.read_pickle(ds['data_path'])
        dirs = os.listdir(ds['model_root'])
        df_models = [x for x in dirs if x.startswith("k")]
        pattern = re.compile(r"k(?P<k>\d+)_a(?P<a>\d+\.\d+)_var(?P<var>\d+\.\d+)")
        for model in df_models:
            m = pattern.match(model)
            k = int(m.group('k'))
            a = float(m.group('a'))
            var = float(m.group('var'))
            if k not in whitelist_k or a not in whitelist_a or var not in whitelist_var:
                continue
            ndocs = sum([int(x) for x in open(os.path.join(ds['model_root'], 'model-seq.dat')).readlines()[1:]])
            analysis = DTMAnalysis(
                ndocs,
                int(model.split("_")[0].split("k")[1]),
                model_root=ds['model_root'],
                model_out_dir=model,
                thesaurus=thesaurus,
                thes_label_col="domain"
            )
            save_dir = os.path.join(ds['model_root'], "analysis")
            analyse(analysis, dataset_df, model, save_dir, coherence=False, document_topic_dist=False, terms=False, plot=True)


if __name__ == "__main__":
    analyse_multi_models()