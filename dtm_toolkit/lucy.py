"""This file will contain text analysis techniques that have drawn on from the paper:

Content Analysis of Textbooks via Natural Language Processing: 
Findings on Gender, Race, and Ethnicity in Texas U.S. History Textbooks,
Lucy et. al (2020)

which have been adapted to conform with our research purposes. These include:
1. coreference resolution of energy technologies in order to count distribution of particular ETs in our corpus
2. dependency parsing of energy keywords to understand how they are described
3. topic modelling of our corpus

We want to be able to perform these tasks and analyse the results:
1. between publications e.g. see how often ETs are talked about by EIA compared to IRENA
    OR compare descriptions of energy technologies between publications 
2. over time e.g. how often are renewable ETs mentioned in 1998 compared to 2018
3. between ETs themselves e.g. how is coal discussed compared to solar (moreso for dependency)
"""

import spacy
# import neuralcoref
import pandas as pd
import os
from collections import Counter
from spacy.matcher import PhraseMatcher
import sys


NUM_CPU = os.cpu_count() - 1 if os.cpu_count() > 1 else 1


def remove_duplicate_matches(matches):
    """
    This function reduces the number of spaCy PhraseMatcher matches by removing matches that occur within another match that have the same id.
    e.g if there is both a match for "oil" and "crude oil" in the same span, then we need to remove the "oil"
    match as the crude oil match takes precedence as it is longer.
    """
    prev_match = None
    reduced_matches = []
    for match in matches:
        if not prev_match:
            reduced_matches.append(match)
        # here we check to see if the matches come from the same match id,
        # if they don't then we don't need to reduce. If they do we may need to reduce
        # by checking the two final elif statements.
        elif prev_match[0] != match[0]:
            reduced_matches.append(match)
        elif match[1] > prev_match[2]:
            reduced_matches.append(match)
        elif (match[1] < prev_match[2]) and (match[2] > prev_match[2]):
            reduced_matches.append(match)
        prev_match = match
    return reduced_matches


class Lucy:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm", disable=['ner','tok2vec'])

    def _load_matcher(self, matcher_keywords):
        """
        Here we load a matcher for the measuring space section of the lucy paper.
        If provided, matcher_keywords takes the form of a dictionary where topic/themes which we want to
        count the number of occurrences within a corpus are the keys and the values are lists of keywords to match
        pertaining to that topic/theme. e.g.

        {
            solar: [photovoltaic, pv, solar power, ...]
        }

        """
        print("loading matcher...")
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LEMMA")
        if matcher_keywords:
            assert type(matcher_keywords) == dict
            for theme, keywords in matcher_keywords.items():
                self.matcher.add(theme, self.nlp.pipe(keywords, n_process=NUM_CPU))
        else:
            for et, keywords in self.energy_clusters.items():
                self.matcher.add(et, self.nlp.pipe(keywords, n_process=NUM_CPU))
        print("loaded matcher...")

    def _sort_energy_clusters(self, reverse):
        for et in self.energy_clusters:
            curr_energy_keywords = self.energy_clusters[et]
            sorted_curr_energy_keywords = sorted([self._preprocess_spacy(t) for t in curr_energy_keywords], key=len, reverse=reverse)
            self.energy_clusters[et] = sorted_curr_energy_keywords

    def load_df(self, df_path):
        if df_path.endswith(".pickle"):
            self.df = pd.read_pickle(df_path)
        elif df_path.endswith(".csv"):
            self.df = pd.read_csv(df_path)
        elif df_path.endswith(".tsv"):
            self.df = pd.read_csv(df_path, sep="\t")
        return self.df
    
    def _save_energy_count_df(self, df, save_path):
        df.to_pickle(save_path)

    def _load_energy_count_df(self, load_path):
        return pd.read_pickle(load_path)

    # def build_coref_df(self, df):
    #     """
    #     Contains strategies found in the 'Measuring Space' section of Lucy et. al (2020)
    #     This function builds a coreferenced dataframe that contains the raw text of each paragraph in a dataframe,
    #     as well as the same paragraph text with all coreferences resolved. For example a raw paragraph such as:
    #     "I like soccer, it is a great game."
    #     Should resolve to
    #     "I like soccer, soccer is a great game."

    #     Args:
    #         df (pd.DataFrame): A dataframe containing the raw paragraph text for coreference resolution

    #     Returns:
    #         df (pd.DataFrame): A dataframe containing two columns, raw and coreference resolved paragraph text as well as paragraph metadata.
    #     """
    #     neuralcoref.add_to_pipe(self.nlp)
    #     coref_docs = []
    #     for doc in self.nlp.pipe(df['para_text'], batch_size=256, n_process=NUM_CPU):
    #         coref_docs.append(doc._.coref_resolved)
    #     # coref_df = pd.DataFrame({
    #     #     "organisation": df['organisation'],
    #     #     "doc_category": df['doc_category'],
    #     #     "filename": df['filename'],
    #     #     "header_text": df['header_text'],
    #     #     "para_text": df['para_text'],
    #     #     "year": df['year'],
    #     #     "coref_para_text": coref_docs
    #     # })
    #     coref_df = df.copy()
    #     coref_df['coref_text'] = coref_docs
    #     coref_df.to_pickle(self.coref_save_path)
    #     return coref_df

    def _identify_energy_descriptors(self, id_, input_para):
        """
        Builds off the strategies found in the 'Identifying Descriptor Words' section of Lucy et. al (2020)
        This function identifies verbs and adjectives associated with energy technology keywords using dependency
        parsing. 
        
        TODO

        Args:
            id_ (int): an arbitrary id for printing purposes, usually corresponds to row number in dataframe
            input_para (str): This is the paragraph to identify energy descriptors in
            # preprocess (boolean): If true, the input_para will be preprocessed, else preprocessing is skipped.
        """
        # find noun chunks in paragraph
        # lemmatise noun chunks
        # match noun chunks with et keywords
        # if match: then find verbs and adjecties that relate to them

        # process paragraph with spacy to return doc
        # check if doc contains ET keywords (maybe in this case, lemma is better)
        # if yes: check if ET keywords are nouns
        # if yes: find adjectives, nouns describing the ET keyword phrase
        # by tracing dependency parse back from the keyword phrase noun.
        pass

    def _reduce_matches(self, matches):
        """
        This function reduces the number of matches by removing matches that occur within another match.
        e.g if there is both a match for "oil" and "crude oil" in the same span, then we need to remove the "oil"
        match as the crude oil match takes precedence as it is longer.
        """
        prev_match = []
        reduced_matches = []
        for match in matches:
            if not prev_match:
                reduced_matches.append(match)
            elif match[1] > prev_match[2]:
                reduced_matches.append(match)
            elif (match[1] < prev_match[2]) and (match[2] > prev_match[2]):
                reduced_matches.append(match)
            prev_match = match
        return reduced_matches

    def _count_energy_occurrences(self, id_, input_para, preprocess=True):
        """
        Builds off the strategies found in the 'Measuring Space' section of Lucy et. al (2020)
        This function counts the prevalence of energy technology keywords in a paragraph of text.

        Args:
            id_ (int): an arbitrary id for printing purposes, usually corresponds to row number in dataframe
            input_para (str): This is the paragraph to count energy occurrences in
            preprocess (boolean): If true, the input_para will be preprocessed, else preprocessing is skipped.

        Returns:
            energy_counts (dict): This is the high level frequency counts for each energy technology
            energy_terms (dict): This is the lower level count of individual energy keywords
            preproc_para (str|None): If preprocessing is true, then this function returns the preprocessed paragraph as well
        """
        print(f"row {id_}")
        preproc_para = None
        energy_counts = Counter()
        energy_terms = Counter()
        if input_para:
            matches = self.matcher(input_para)
            matches = remove_duplicate_matches(matches)
            for match_id, start, end in matches:
                et = self.nlp.vocab.strings[match_id]
                term = input_para[start:end].text.lower()
                energy_counts[et] += 1
                energy_terms[term] += 1
        return energy_counts, energy_terms, preproc_para
    
    def _preprocess_spacy(self, text):
        lemma_doc = ""
        text = text.lower()
        # lemmatise
        doc = self.nlp(text)
        for tok in doc:
            if tok.pos_ == "PART":
                lemma_doc = lemma_doc + tok.whitespace_
            lemma_doc = lemma_doc + tok.lemma_ + tok.whitespace_
        return lemma_doc

    def get_energy_count_by_x(self, df, groupby="doc_category", column="coref_energy_counts"):
        """
        Contains strategies found in the 'Measuring Space' section of Lucy et. al (2020)
        This function counts the frequency of energy technologies in paragraphs by a specified groupby clause either with coreference resolved
        paragraphs, or raw paragraphs. By default groupby is on publication.

        Each paragraph count needs to be in the form as created by:  _count_energy_occurrences
        In theory, can be used for any Counter based 
        i.e. {"coal": 4, "bio": 3, "oil": 2, ...}

        Args:
            df (pd.DataFrame): dataframe containing energy counts in each paragraph
            column (str): the count column to use either "raw_energy_counts" or "coref_energy_counts"

        Returns:
            df (pd.DataFrame): dataframe containing enriched rows with counts of each energy technology
        """
        return df.groupby(groupby)[column].aggregate(lambda x: sum(x, Counter()))
    
    def run_measuring_space(self, df, para_col_name, matcher_keywords=None, save_col_prefix="lucy"):
        """
        This function is responsible for running the core logic identified in the "Measuring Space"
        section of the Lucy et. al (2020) paper. We identify how much 'space' (frequency counts) is allocated
        to different groups of energy technologies by identifying energy-specific keywords within our corpus.

        Args:
            df (pd.DataFrame): The dataframe containing some paragraph text. This could be coref resolved paragraph text,
                similar to Lucy, or else just raw paragraph text.
            preprocess (boolean): This flag dictates whether the paragraphs passed in need to be preprocessed (True) or not.
            type_ (str): This variable dictates which column of the dataframe to pull the paragraphs from. One can either choose
                "coref" which will grab paragraphs from a column called "coref_para_text" or any other string will take paragraphs from
                "para_text".
        
        Returns:
            count_df (pd.DataFrame): Same as the dataframe passed in to this function, except has been enriched with counts of each energy technology
                within each paragraph, as well as counts of terms that were matched within each paragraph. If preprocess is true, will also add the preprocessed
                paragraph text to the dataframe under the "preproc_para" column.
        """

        counts = []
        terms = []
        preproc_paras = []
        count_df = df.copy()
        self._load_matcher(matcher_keywords)
        # ensure no nonetype terms
        df[para_col_name] = df[para_col_name].fillna(value="")
        for i, doc in enumerate(self.nlp.pipe(df[para_col_name], n_process=11, batch_size=256)):
            # print(i)
            e_counts, e_terms, preproc_para = self._count_energy_occurrences(i, doc)
            counts.append(e_counts)
            terms.append(e_terms)
            preproc_paras.append(preproc_para)
        count_df[save_col_prefix + "counts"] = counts
        count_df[save_col_prefix + "terms"] = terms
        return count_df


def run_measuring_space(df, para_col_name, matcher_keywords, save_path=None, save=True, save_col_prefix="lucy_"):
    """ This function is responsible for initialising the Lucy class to run the 
        core logic identified in the "Measuring Space" section of the Lucy et. al (2020) paper: 
        'Content Analysis of Textbooks via Natural Language Processing: Findings on Gender, Race, 
        and Ethnicity in Texas U.S. History Textbooks.'

    Args:
        df (pd.DataFrame): dataframe of the corpus to compute frequency counts on.
        para_col_name (string): the column of the dataframe containing the textual content to analyse.
        matcher_keywords (dict): the key-value mapping between a theme and a list of keywords that pertain to the theme.
            For example, to discover how frequently energy technologies are discussed in a corpus of energy documents, we can use
            matcher_keywords like this:
    {
    "coal": [
        "hele",
        "supercritical",
        "coal",
		"coal seam",
        "coal gasification",
        "coal industry",
        "coal mine",
        ],

    "bio": [
        "bio",
        "biomass",
        "biofuel",
        "ethanol",
        "biodiesel",
        "methanol",
        ],
    "oil": [
        "oil",
        "crude oil",
        "diesel oil",
        "domestic fuel oil",
        "oil technology",
        "oil-burning power station",
        "oilfield",
        "petroleum oil"
        ]
    }
        save_path (string): path where to save the enriched dataframe
        save_col_prefix (str, optional): column prefix for the frequency counts of particular terms within labels and also for labels. 
            Defaults to "lucy_".
    """
    l = Lucy()
    ## neuralcoref bugged so disabled ##
    # coref_df = l.build_coref_df(df)
    lucy_df = l.run_measuring_space(df, para_col_name, matcher_keywords, save_col_prefix)
    if save and not save_path:
        print("need to provide save path")
        sys.exit(1)
    elif save:
        lucy_df.to_csv(save_path)
    return lucy_df