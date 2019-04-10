from gensim.models import KeyedVectors
import numpy as np

import re
import logging
import random

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DATA_LOADER")

def preprocess(text):
    #EMOJIS
    text = re.sub(r":\)","emojihappy1",text)
    text = re.sub(r":P","emojihappy2",text)
    text = re.sub(r":p","emojihappy3",text)
    text = re.sub(r":>","emojihappy4",text)
    text = re.sub(r":3","emojihappy5",text)
    text = re.sub(r":D","emojihappy6",text)
    text = re.sub(r" XD ","emojihappy7",text)
    text = re.sub(r" <3 ","emojihappy8",text)

    text = re.sub(r":\(","emojisad9",text)
    text = re.sub(r":<","emojisad10",text)
    text = re.sub(r":<","emojisad11",text)
    text = re.sub(r">:\(","emojisad12",text)

    #MENTIONS "(@)\w+"
    text = re.sub(r"(@)\w+","mentiontoken",text)
    
    #WEBSITES
    text = re.sub(r"http(s)*:(\S)*","linktoken",text)

    #STRANGE UNICODE \x...
    text = re.sub(r"\\x(\S)*","",text)

    #General Cleanup and Symbols
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
    text = re.sub(r"\'s", " \'s", text)
    text = re.sub(r"\'ve", " \'ve", text)
    text = re.sub(r"n\'t", " n\'t", text)
    text = re.sub(r"\'re", " \'re", text)
    text = re.sub(r"\'d", " \'d", text)
    text = re.sub(r"\'ll", " \'ll", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\(", " \( ", text)
    text = re.sub(r"\)", " \) ", text)
    text = re.sub(r"\?", " \? ", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip().lower()

def load_glove():
    vocab = []
    embd = []
    file = open(config.GLOVE_WORD2VEC, "r")
    for line in file:
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    file.close()
    return vocab, embd

def seperate_data(inf, tp="train"):
    if tp=='train':
        pos_out = open(config.POS_FILE, "a")
        neg_out = open(config.NEG_FILE, "a")
        neu_out = open(config.NEU_FILE, "a")
    if tp=='test':
        pos_out = open(config.POS_TEST_FILE, "a")
        neg_out = open(config.NEG_TEST_FILE, "a")
        neu_out = open(config.NEU_TEST_FILE, "a")
    seen = 0
    with open(inf, "r") as f:
        for line in f:
            seen += 1
            id, lbl, sentence = line.strip().split("\t", 3)
            if lbl == "positive":
                pos_out.write(sentence + "\n")
            if lbl == "negative":
                neg_out.write(sentence + "\n")
            if lbl == "neutral":
                neu_out.write(sentence + "\n")
            if (seen % 1000 == 0):
                logger.info("{seen} loaded".format(seen=seen))
    pos_out.close()
    neg_out.close()
    neu_out.close()

def get_dataset(posfile, negfile, neufile, randomize=True):
    pos_x = [ s.strip() for s in list(open(posfile, "r").readlines()) ]
    neg_x = [ s.strip() for s in list(open(negfile, "r").readlines()) ]
    neu_x = [ s.strip() for s in list(open(neufile, "r").readlines()) ]
    if randomize:
        random.shuffle(pos_x)
        random.shuffle(neg_x)
        random.shuffle(neu_x)

    x = pos_x + neg_x + neu_x
    x = [ preprocess(s) for s in x ]

    pos_label = [ [1, 0, 0] for _ in pos_x ]
    neg_label = [ [0, 1, 0] for _ in neg_x ]
    neu_label = [ [0, 0, 1] for _ in neu_x ]
    y = np.concatenate([pos_label, neg_label, neu_label], 0)

    return [x, y]

def get_batch(data, batch_size, num_epochs, shuffle=True):
    # generate batch iterator for a dataset
    data = np.array(data)
    data_size = len(data)
    num_batchs_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batchs_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

if __name__ == '__main__':
    # seperate_data(config.TRAIN_TPL.format(year=2013))
    # seperate_data(config.TRAIN_TPL.format(year=2015))
    # seperate_data(config.TRAIN_TPL.format(year=2016))
    seperate_data("dataset/SemEval2017-task4-test/SemEval2017-task4-test.subtask-A.english.txt", tp='test')
    # x, y = get_dataset(config.POS_FILE, config.NEG_FILE, config.NEU_FILE)
    # print(x.shape)
    # print(y.shape)
    # v, e = load_glove()
    # print(len(v))
    # print(len(e))