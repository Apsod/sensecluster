import os
import collections
import itertools
import pickle
import logging
import argparse

import numpy
import scipy
import sklearn
import sklearn.cluster
import sklearn.metrics
import tqdm


LANGS = ['english', 'german', 'swedish', 'latin']
Data = collections.namedtuple('Data', ['X', 'W', 'C', 'words'])

def read_emb(language, corpus):
    words = []
    embs = []
    for line in open('{}_{}.emb'.format(language, corpus), 'rt'):
        word, emb = line.strip().split('\t')
        emb = [float(x) for x in emb.split()]
        words.append(word)
        embs.append(emb)

    return words, embs

def load(language):
    logging.info('loading {}'.format(lang))
    w1, e1 = read_emb(language, 1)
    w2, e2 = read_emb(language, 2)
    
    C = numpy.array([1 for _ in w1] + [2 for _ in w2])
    W = numpy.array(w1 + w2)
    X = numpy.array(e1 + e2)
    unique = list(set((w1 + w2)))

    logging.info('C: {}, W: {}, X: {}'.format(C.shape, W.shape, X.shape))

    keep = ~numpy.isnan(X).any(1)
    logging.warning("dropping {} nan-embeddings".format(sum(~keep)))
    return Data(X[keep], W[keep], C[keep], words=unique)

def jsd(p, q):
    p = p / p.sum()
    q = q / q.sum()
    m = (p + q) / 2
    return (scipy.stats.entropy(p,m) + scipy.stats.entropy(q, m)) / 2

def cluster(data, word, N=8, k=2, n=5):
    model = sklearn.cluster.KMeans(N)
    x = data.X[data.W == word]
    c = data.C[data.W == word]

    labels = model.fit_predict(x)

    c1 = numpy.zeros(N)
    c2 = numpy.zeros(N)
    for l in labels[c==1]:
        c1[l] += 1.0
    for l in labels[c==2]:
        c2[l] += 1.0

    t1 = (
            ((c1 <= k) & (c2 >= n)).any() |
            ((c2 <= k) & (c1 >= n)).any() 
    )
    t2 = jsd(c1, c2)
    return t1, t2

def cluster_all_targets(language, N=8):
    logging.info('clustering {}'.format(lang))
    data = load(language)
    results = []
    for w in tqdm.tqdm(data.words):
        results.append((w, *cluster(data, w)))
    return results

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    for lang in LANGS:
        res = cluster_all_targets(lang)
        path1 = 'answer/task1/{}.txt'.format(lang)
        path2 = 'answer/task2/{}.txt'.format(lang)
        os.makedirs(os.path.dirname(path1), exist_ok=True)
        os.makedirs(os.path.dirname(path2), exist_ok=True)
        with open(path1, 'wt') as handle1, open(path2, 'wt') as handle2:
            for (w, t1, t2) in res:
                handle1.write('{}\t{}\n'.format(w, 1 if t1 else 0))
                handle2.write('{}\t{}\n'.format(w, t2))


