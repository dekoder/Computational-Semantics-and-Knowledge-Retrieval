#!/usr/local/bin/python3

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from scipy import stats

wordlist = []
scores = []
print("Loading ic")
brown_ic = wordnet_ic.ic('ic-brown.dat')
semcor_ic = wordnet_ic.ic('ic-semcor.dat')
plots = []

def load_word_list():
    PATH = "./datas/MTURK-771.csv"
    with open(PATH) as f:
        for line in f:
            w1, w2, s = line.strip().split(",")
            wordlist.append((w1, w2))
            scores.append(float(s))

def path_sim():
    PATH = "./datas/path_similarity.csv"
    outf = open(PATH, "w")
    result = []
    for pair in wordlist:
        # print(pair)
        w1s = wn.synsets(pair[0])
        w2s = wn.synsets(pair[1])
        s = 0.0
        cnt = 0
        for w1 in w1s:
            for w2 in w2s:
                tmp = wn.path_similarity(w1, w2)
                if not tmp:
                    continue
                if tmp > s:
                    s = tmp
                    cnt += 1
        s /= cnt
        # print(s)
        outf.write("{}\n".format(s))
        result.append(s)
    outf.close()
    return (stats.spearmanr(result, scores))

def wup_sim():
    PATH = "./datas/wup_similarity.csv"
    outf = open(PATH, "w")
    result = []
    for pair in wordlist:
        # print(pair)
        w1s = wn.synsets(pair[0])
        w2s = wn.synsets(pair[1])
        s = 0.0
        cnt = 0
        for w1 in w1s:
            for w2 in w2s:
                tmp = wn.wup_similarity(w1, w2)
                if not tmp:
                    continue
                if tmp > s:
                    s = tmp
                    cnt += 1
        s /= cnt
        # print(s)
        outf.write("{}\n".format(s))
        result.append(s)
    outf.close()
    return (stats.spearmanr(result, scores))

def res_similarity(ic, name):
    PATH = "./datas/res_similarity_{name}.csv".format(name=name)
    outf = open(PATH, "w")
    result = []
    for pair in wordlist:
        # print(pair)
        w1s = wn.synsets(pair[0])
        w2s = wn.synsets(pair[1])
        s = 0.0
        cnt = 0
        for w1 in w1s:
            for w2 in w2s:
                try:
                    tmp = w1.res_similarity(w2, ic)
                    if tmp == None:
                        continue
                    if tmp > s:
                        s = tmp
                        cnt += 1
                except Exception as e:
                    pass
        if cnt > 0:
            s /= cnt
        # print(s)
        outf.write("{}\n".format(s))
        result.append(s)
    outf.close()
    return (stats.spearmanr(result, scores))

def jcn_similarity(ic, name):
    PATH = "./datas/jcn_similarity_{name}.csv".format(name=name)
    outf = open(PATH, "w")
    result = []
    for pair in wordlist:
        # print(pair)
        w1s = wn.synsets(pair[0])
        w2s = wn.synsets(pair[1])
        s = 0.0
        cnt = 0
        for w1 in w1s:
            for w2 in w2s:
                try:
                    tmp = w1.jcn_similarity(w2, ic)
                    if tmp == None:
                        continue
                    if tmp > s:
                        s = tmp
                        cnt += 1
                except Exception as e:
                    pass
        if cnt > 0:
            s /= cnt
        # print(s)
        outf.write("{}\n".format(s))
        result.append(s)
    outf.close()
    return (stats.spearmanr(result, scores))

def lin_similarity(ic, name):
    PATH = "./datas/lin_similarity_{name}.csv".format(name=name)
    outf = open(PATH, "w")
    result = []
    for pair in wordlist:
        # print(pair)
        w1s = wn.synsets(pair[0])
        w2s = wn.synsets(pair[1])
        s = 0.0
        cnt = 0
        for w1 in w1s:
            for w2 in w2s:
                try:
                    tmp = w1.lin_similarity(w2, ic)
                    if tmp == None:
                        continue
                    if tmp > s:
                        s = tmp
                        cnt += 1
                except Exception as e:
                    pass
        if cnt > 0:
            s /= cnt
        # print(s)
        outf.write("{}\n".format(s))
        result.append(s)
    outf.close()
    return (stats.spearmanr(result, scores))

# Trained by google news
def google_vec():
    fn = "./big-corpus/GoogleNews-vectors-negative300.bin"
    PATH = "./datas/word2vec-google.csv"
    outf = open(PATH, "w")
    result = []
    model = KeyedVectors.load_word2vec_format(fn, binary=True)
    for pair in wordlist:
        s = model.similarity(pair[0], pair[1])
        result.append(s)
        outf.write("{}\n".format(s))
    outf.close()
    return (stats.spearmanr(result, scores))

def glove(transform=False):
    fn = "./big-corpus/glove.6B/glove.6B.300d.txt"
    vec_file = "./big-corpus/glove.6B/glove.6B.300d.word2vec"
    if transform:
        glove2word2vec(fn, vec_file)
    PATH = "./datas/word2vec-glove.csv"
    outf = open(PATH, "w")
    result = []
    model = KeyedVectors.load_word2vec_format(vec_file, binary=False)
    for pair in wordlist:
        s = model.similarity(pair[0], pair[1])
        result.append(s)
        outf.write("{}\n".format(s))
    outf.close()
    return (stats.spearmanr(result, scores))

if __name__ == "__main__":
    load_word_list()
    path_sim()
    wup_sim()
    res_similarity(brown_ic, "brown")
    res_similarity(semcor_ic, "semcor")
    jcn_similarity(brown_ic, "brown")
    jcn_similarity(semcor_ic, "semcor")
    lin_similarity(brown_ic, "brown")
    lin_similarity(semcor_ic, "semcor")
    google_vec()
    glove()