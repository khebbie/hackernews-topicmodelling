#!/usr/bin/env python

# LDA: generic LDA implementation using gensim

import logging
from gensim import corpora, models, similarities
import nltk
import argparse
import csv
import os

class LDA():
    def __init__(self, __logLevel, __topics):
        self.log = logging.getLogger('LDA')
        self.log.setLevel(__logLevel)
        self.topics = __topics
        self.documents = []

        # 1. Load text file
        self.readFiles()
        self.log.debug(self.documents)

        # 2. Compute LDA
        self.computeLDA()

    def readFile(self, filename):
        f = open(filename, 'r')
        text = f.read()
        print("file content:")
        print(text)
        self.documents.append(unicode(text, 'utf-8'))


    def readFiles(self):
        for filename in os.listdir('./content'):
            self.readFile('content/' + filename)

    def computeLDA(self):
        utf_stopwords = ['\u0xc2']
        tokenizer = nltk.tokenize.RegexpTokenizer('\(.*\)|[\s\.\,\%\:\$]+', gaps=True)
        texts = [[word for word in tokenizer.tokenize(document.lower()) if word not in nltk.corpus.stopwords.words('english')] for document in self.documents]
        self.log.debug(texts)

        # remove words that appear only once
        all_tokens = sum(texts, [])
        tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
        texts = [[word for word in text if word not in tokens_once] for text in texts]
        self.log.debug(texts)

        self.dictionary = corpora.Dictionary(texts)
        self.dictionary.save('/tmp/deerwester.dict')
        self.log.debug(self.dictionary)

        self.corpus = [self.dictionary.doc2bow(text) for text in texts]
        corpora.MmCorpus.serialize('/tmp/deerwester.mm', self.corpus)
        self.log.debug(self.corpus)

        self.tfidf = models.TfidfModel(self.corpus)
        corpus_tfidf = self.tfidf[self.corpus]
        self.log.debug(self.tfidf)

        lda = models.ldamodel.LdaModel(corpus=self.corpus, id2word=self.dictionary, num_topics=self.topics, update_every=1, chunksize=10000, passes=20)
        lda.print_topics(10)

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Simple LDA using gensim")
    parser.add_argument('--verbose', '-v',
            help = "Be verbose -- debug logging level",
            required = False,
            action = 'store_true')
    parser.add_argument('--topics', '-t',
            help = "Number of topics",
            required = True)
    args = parser.parse_args()

    # Logging
    logLevel = logging.INFO
    if args.verbose:
        logLevel = logging.DEBUG
    logging.basicConfig(level=logLevel)
    logging.info('Initializing...')

    # Instance
    lda = LDA(logLevel, int(args.topics))

    logging.info('Done.')
