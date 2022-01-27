import os

NGRAM_PATH = os.path.join(os.path.dirname(__file__), "NGRAMS_PREPROC.txt")

def get_ngrams():
    return [" %s " % x.strip('\n') for x in sorted([y for y in open(NGRAM_PATH, "r").readlines()], key=lambda x: len(x), reverse=True)]