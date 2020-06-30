import gzip
import os
import logging
import itertools
import argparse
import multiprocessing

import torch
import numpy

def get_ctxs(lang, corpus, pad_token):
    ret = []
    for line in open('{}_{}.ctx'.format(lang, corpus), 'rt'):
        token, start, stop, ixs = line.strip().split('\t')
        start = int(start)
        stop = int(stop)
        ixs = [int(ix) for ix in ixs.split()]

        free = len(ixs) - stop + start

        lb = max(0, start - free // 2)
        rb = min(512, stop + free // 2)

        arr = numpy.full(512, pad_token)

        arr[:(rb - ub)] = ixs[lb:rb]
        

        ret.append(token, start-lb, stop-lb, arr)

    
def chunk(gen, chunksize=100):
    chnk = list(itertools.islice(gen, chunksize))
    while chnk:
        yield chnk
        chnk = list(itertools.islice(gen, chunksize))


def get_ctxs(lang, corpus, pad_token):
    for line in open('{}_{}.ctx'.format(lang, corpus), 'rt'):
        token, start, stop, ixs = line.strip().split('\t')
        start = int(start)
        stop = int(stop)
        ixs = [int(ix) for ix in ixs.split()]

        free = len(ixs) - stop + start

        lb = max(0, start - free // 2)
        rb = min(512, stop + free // 2)
        ixs = ixs[lb:rb]
        arr = numpy.full(512, pad_token)
        arr[:len(ixs)] = ixs
        yield token, start-lb, stop-lb, arr


if __name__ == '__main__':
    with torch.no_grad():
        logging.basicConfig(level=logging.DEBUG)
        logging.info('loading XLM-R')
        xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.large').eval().cuda()
        logging.info('finding contexts')
        for language, corpus in itertools.product(
                ['english', 'german', 'swedish', 'latin'],
                [1,2]
                ):
            logging.info('Running {} {}'.format(language, corpus))
            with open('{}_{}.emb'.format(language, corpus), 'wt') as handle:
                for chnk in chunk(get_ctxs(language, corpus, xlmr.task.dictionary.pad_index)):
                    tokens, starts, stops, seqs = zip(*chnk)
                    seqs = torch.tensor(seqs)
                    embs = xlmr.extract_features(seqs)# [0][start:stop].mean(0).tolist()
                    for token, start, stop, emb in zip(tokens, starts, stops, embs):
                        emb_txt = ' '.join(['{:.4e}'.format(x) for x in emb[start:stop].mean(0)])
                        handle.write('{}\t{}\n'.format(token, emb_txt))

