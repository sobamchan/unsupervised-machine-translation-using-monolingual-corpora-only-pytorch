import random


non_drop_list = ['</s>', '.']


def shuffle_order(words, k=3):
    '''
    words: list of words
    k: int from "unsupervised machine translation
                 using monolingual corpora only"
    '''
    word_n = len(words)
    idxs = list(range(word_n))
    new_word2idx = {}
    for idx, word in zip(idxs, words):
        idx_min = max([0, idx - k])
        idx_max = min([word_n, idx + k])
        while True:
            new_idx = random.choice(idxs[idx_min:idx_max])
            if new_idx not in new_word2idx.keys():
                new_word2idx[word] = new_idx
                break
    new_words = sorted(new_word2idx.items(), key=lambda x: x[1])
    new_words = [nw[0] for nw in new_words]
    return new_words


def drop_words(words, p=0.1):
    '''
    words: list of words
    p: dropout ratio
    '''
    new_words = []
    for word in words:
        if random.random() < p and word not in non_drop_list:
            continue
        new_words.append(word)
    return new_words


def run(s_words, p=0.1, k=3):
    words = s_words.split()
    return ' '.join(shuffle_order(drop_words(words)))
