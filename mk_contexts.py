import gzip
import os
import logging
import itertools
import argparse
import multiprocessing

import torch



CORPUS = dict(
    english = {1: 'ccoha1.txt.gz', 2: 'ccoha2.txt.gz'},
    german = {1: 'dta.txt.gz', 2: 'bznd.txt.gz'},
    latin = {1: 'LatinISE1.txt.gz', 2: 'LatinISE2.txt.gz'},
    swedish = {1: 'kubhist2a.txt.gz', 2: 'kubhist2b.txt.gz'}
)


def get_corpus(dataroot, language, corpus, which='lemma'):
    assert language in CORPUS
    assert corpus in [1, 2]
    assert which in ['lemma', 'token']
    path = os.path.join(
            dataroot, 
            language,
            'corpus{}'.format(corpus), 
            which, 
            CORPUS[language][corpus])
    
    yield from gzip.open(path, 'rt')

def get_targets(dataroot, language):
    assert language in CORPUS

    path = os.path.join(
            dataroot,
            language,
            'targets.txt')
    
    words = set()
    for line in open(path):
        words.add(line.strip())

    return words

def find_sublist(sub, seq):
    N = len(seq)
    M = len(sub)
    for i in range(0, N-M):
        if seq[i:i+M] == sub:
            yield (i, i+M)


def get_contexts(dataroot, language, corpus, model):
    targets = get_targets(dataroot, language)

    t2e = {t: model.encode(t).tolist()[1:-1] for t in targets}

    def get_locs(tokens):
        for target, encoding in t2e.items():
            for loc in find_sublist(encoding, tokens):
                yield target, loc

    for line in get_corpus(dataroot, language, corpus):
        tokens = model.encode(line.strip())
        for target, loc in get_locs(tokens.tolist()):
            yield target, loc, tokens



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
            'dataroot', 
            type=str,
            help='/path/to/test_data_public/',
            )

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logging.info('loading XLM-R')
    xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.large').eval()
    logging.info('finding contexts')

    def f(language_corpus):
        language = language_corpus[0]
        corpus = language_corpus[1]
        logging.info('writing contexts: {} {}'.format(language, corpus))
        with open('{}_{}.ctx'.format(language, corpus), 'wt') as handle:
            for target, loc, tokens in get_contexts(args.dataroot, language, corpus, xlmr):
                token_txt = ' '.join(map(str, tokens.tolist()))
                handle.write('{}\t{}\t{}\t{}\n'.format(target, loc[0], loc[1], token_txt))

        return language, corpus
    
    with multiprocessing.Pool(8) as p:
        for language, corpus in p.imap_unordered(f, itertools.product(CORPUS, [1,2])):
            logging.info('DONE: {} {}'.format(language, corpus))

