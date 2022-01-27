from spacy.lang.en.stop_words import STOP_WORDS
import re
import os
import spacy

DEFAULT_PROC_NGRAM_PATH = os.path.join(os.path.dirname(__file__), "NGRAMS_PREPROC.txt")
DEFAULT_UNPROC_NGRAM_PATH = os.path.join(os.path.dirname(__file__), "NGRAMS_LG.txt")

class Preprocessing:
    """This is a helper class for preprocessing a list of spaCy objects. The preprocessing pipeline is
    described in self.preprocess()
    """

    def __init__(self, docs, term_blacklist=[], verbose=True, ngram_file_path=DEFAULT_PROC_NGRAM_PATH):
        """Class initialise function.

        Args:
            docs ([spacy.doc.Doc]): list of spacy documents to preprocess
            ngram_file_path (str): path to ngram file. when an ngram is found in a document, it is joined with an _. 
                The ngram file must be a newline separated file of ngrams. see DEFAULT_NGRAM_PATH for example. Defaults to DEFAULT_NGRAM_PATH
            term_blacklist ([str], optional): list of tokens that should be removed from any document they exist in. Defaults to [].
            verbose (bool, optional): whether or not to print progress indicators. Defaults to True.
        """
        self.docs = docs
        self.term_blacklist = term_blacklist
        self.verbose = verbose
        self.preproc_ngram_path = ngram_file_path
    
    def _print_progress(self, n, total, increment=0.25):
        if n == int(total)*0.25:
            print("25%...")
        elif n == int(total)*0.5:
            print("50%...")
        elif n == int(total)*0.75:
            print("75%...")

    def _substitute_ngrams(self, doc, ngrams):
        ngrammed_doc = []
        for sent in doc:
            ngrammed_sent = " ".join(sent)
            for ngram_patt in ngrams:
                ngrammed_sent = ngrammed_sent.replace(ngram_patt, f' {ngram_patt.strip().replace(" ", "_")} ')
            # edge cases are start and end of sent or sent is whole ngram
            for ngram_patt in ngrams:
                ngram = ngram_patt.strip()
                if ngrammed_sent.startswith(ngram):
                    ngrammed_sent = re.sub(r'^%s ' % ngram, r'%s ' % ngram.replace(" ", "_"), ngrammed_sent)
                if ngrammed_sent.endswith(ngram):
                    ngrammed_sent = re.sub(r' %s$' % ngram, r' %s' % ngram.replace(" ", "_"), ngrammed_sent)
                if ngrammed_sent == ngram:
                    ngrammed_sent = ngrammed_sent.replace(" ", "_")
                    break
            split_sent = ngrammed_sent.split(" ")
            ngrammed_doc.append(split_sent)
        return ngrammed_doc

    def add_ngrams(self):
        """
        ['here', 'is', 'new', 'south', 'wales']
        match with ngram 'new south wales'
        """
        ngram_strings = [" %s " % x.strip('\n') for x in sorted([y for y in open(self.preproc_ngram_path, "r").readlines()], key=lambda x: len(x), reverse=True)]
        ngrammed_docs = []
        for i,doc in enumerate(self.paras_processed):
            if self.verbose:
                print(f"doc {i}")
                self._print_progress(i, len(self.paras_processed))
            ngrammed_docs.append(self._substitute_ngrams(doc, ngram_strings))
        self.paras_processed = ngrammed_docs
        if self.verbose:
            print("done ngramming!")

    def get_merged_sents(self):
        """
        Once preprocessed, this function returns all the tokens in a doc in one list.

        e.g. [['this', 'is', 'the', 'first', 'sentence'], ['second', 'sentence']]

        becomes:

        ['this', 'is', 'the', 'first', 'sentence', 'second', 'sentence']
        """
        if self.paras_processed:
            corpus = []
            for doc in self.paras_processed:
                merged_doc = []
                for sent in doc:
                    merged_doc.extend(sent)
                corpus.append(merged_doc)
        return corpus

    def get_merged_docs(self, keep_empty=False):
        """
        Once preprocessed, this function returns a string of space-separated tokens for a document.

        e.g. [['this', 'is', 'the', 'first', 'sentence'], ['second', 'sentence']]

        becomes:

        'this is the first sentence second sentence'
        """
        if self.paras_processed:
            merged_docs = self.get_merged_sents()
            if keep_empty:
                return [" ".join(doc) for doc in merged_docs]
            else:
                return [" ".join(doc) for doc in merged_docs if doc != [] and doc != [""]]

    def preprocess(
            self,
            ngrams=True
        ):
        """This function takes the spaCy documents found in this classes docs attribute and preprocesses them.
        The preprocessing pipeline tokenises each document and removes:
        1. punctuation
        2. spaces
        3. numbers
        4. urls
        5. stop words and single character words.

        It then lemmatises and lowercases each token and joins multi-word tokens together with an _.
        It then adds ngrams from a ngram list by joining matched ngrams in the corpus with an _.

        Args:
            ngrams (bool, optional): Whether to add ngrams or to keep the corpus as unigram. Defaults to True.
        """
        self.paras_processed = []
        for doc in self.docs:
            sents = []
            for s in doc.sents:
                words = []
                for w in s:
                    # PREPROCESS: lemmatize
                    # PREPROCESS: remove * puncuation
                    #                    * words that are / contain numbers
                    #                    * URLs
                    #                    * stopwords
                    #                    * words of length==1
                    if not w.is_punct \
                        and not w.is_space \
                        and not w.like_num \
                        and not any(i.isdigit() for i in w.lemma_) \
                        and not w.like_url \
                        and not w.text.lower() in STOP_WORDS \
                        and len(w.lemma_) > 1:
                        words.append(w.lemma_.lower().replace(" ", "_"))
                sents.append(words)
            self.paras_processed.append(sents)
        if ngrams:
            if self.verbose:
                print("adding ngrams...")
            self.matches = self.add_ngrams()
            # self._add_bigrams(ngrams=ngrams)
        new_paras = []
        if self.verbose:
            print("filtering terms...")
        for i,doc in enumerate(self.paras_processed):
            if self.verbose:
                self._print_progress(i, len(self.paras_processed))
            filtered_doc = []
            for sent in doc:
                filtered_doc.append([w for w in filter(lambda x: x not in self.term_blacklist and x != '', sent)])
            new_paras.append(filtered_doc)
        self.paras_processed = new_paras
        return self.paras_processed


class NgramPreprocessing(Preprocessing):
    """Class for generating preprocessing ngram list as seen in NGRAMS_PREPROC.txt

    Args:
        Preprocessing (Class): Parent class which holds the preprocessing logic.
    """

    def __init__(self, unproc_ngram_path=DEFAULT_UNPROC_NGRAM_PATH):
        """Initialisation function. Takes in a path to ngrams that need to be preprocessed. File should be a
        newline separated list of ngrams. see DEFAULT_UNPROC_NGRAM_PATH for an example.

        Args:
            unproc_ngram_path (str): path where to find newline separated unprocessed ngrams.
        """
        self.nlp = spacy.load('en_core_web_sm')
        self.ngram_unprocessed_path = unproc_ngram_path
        self.docs = self.nlp.pipe([x.strip('\n') for x in open(self.ngram_unprocessed_path).readlines()], n_process=11, batch_size=128)
    
    def preprocess(self, save_path=None, keep_empty=False):
        super().preprocess(ngrams=False)
        corpus = self.get_merged_docs(keep_empty=keep_empty)
        if save_path:
            with open(os.path.join(save_path), "w+") as fp:
                for phrase in corpus:
                    fp.write(f"{' '.join(phrase)}\n")
        else:
            return corpus


if __name__ == "__main__":
    ngrampp = NgramPreprocessing()
    ngrampp.preprocess()