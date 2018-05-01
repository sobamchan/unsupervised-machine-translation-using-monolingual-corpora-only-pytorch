import random
import unicodedata
from collections import Counter
import re
import torch
from torch.autograd import Variable
from torch import LongTensor as LT


def flatten(li):
    return [item for subli in li for item in subli]


def get_batch(batch_size, train_data):
    random.shuffle(train_data)
    sind = 0
    eind = batch_size
    while eind < len(train_data):
        batch = train_data[sind:eind]
        tmp = eind
        eind = eind + batch_size
        sind = tmp
        yield batch

    if eind >= len(train_data):
        batch = train_data[sind:]
        yield batch


def pad_to_batch(batch, xw2i, yw2i):
    src, tgt = batch['src'], batch['tgt']
    batch = list(zip(src, tgt))
    sorted_batch = sorted(batch,
                          key=lambda b: b[0].size(1),
                          reverse=True)
    x, y = list(zip(*sorted_batch))
    max_x = max([s.size(1) for s in x])
    max_y = max([s.size(1) for s in y])
    x_p, y_p = [], []
    for i in range(len(batch)):
        if x[i].size(1) < max_x:
            x_p.append(torch.cat([x[i],
                                 Variable(LT([xw2i['<PAD>']]
                                          * (max_x - x[i].size(1))))
                                 .view(1, -1)], 1))
        else:
            x_p.append(x[i])

        if y[i].size(1) < max_y:
            y_p.append(torch.cat([y[i],
                                 Variable(LT([yw2i['<PAD>']]
                                          * (max_y - y[i].size(1))))
                                 .view(1, -1)], 1))
        else:
            y_p.append(y[i])

    input_var = torch.cat(x_p)
    target_var = torch.cat(y_p)
    input_len = [list(map(lambda s: s == 0, t.data)).count(False)
                 for t in input_var]
    target_len = [list(map(lambda s: s == 0, t.data)).count(False)
                  for t in target_var]

    return input_var, target_var, input_len, target_len


def prepare_sequence(seq, w2i):
    idxs = list(map(lambda w: w2i[w]
                if w2i.get(w) is not None else w2i['<UNK>'], seq))
    return Variable(LT(idxs))


def prepare_batch(batch, sw2i, tw2i):
    new_batch = {'src': [], 'tgt': []}
    for s, t in zip(batch['src'], batch['tgt']):
        s_p = prepare_sequence(s.split(), sw2i).view(1, -1)
        t_p = prepare_sequence(t.split(), tw2i).view(1, -1)
        new_batch['src'].append(s_p)
        new_batch['tgt'].append(t_p)
    return new_batch


def unicode_to_ascii(s):
    return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            )


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([,.!?])", r" \1 ", s)
    s = re.sub(r"[^a-zA-Z,.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def get_dataset(src_path, tgt_path):
    src_sents = open(src_path, 'r', encoding='utf-8').readlines()
    tgt_sents = open(tgt_path, 'r', encoding='utf-8').readlines()

    X_r = [s.split() for s in src_sents]
    Y_r = [s.split() for s in tgt_sents]
    src_vocab = Counter(list(set(flatten(X_r))))
    src_vocab = [cnt[0] for cnt in src_vocab.most_common(30000)]
    tgt_vocab = Counter(list(set(flatten(Y_r))))
    tgt_vocab = [cnt[0] for cnt in tgt_vocab.most_common(30000)]

    sw2i = {'<PAD>': 0, '<UNK>': 1, '<s>': 2, '</s>': 3}
    for vo in src_vocab:
        if sw2i.get(vo) is None:
            sw2i[vo] = len(sw2i)
    si2w = {v: k for k, v in sw2i.items()}

    tw2i = {'<PAD>': 0, '<UNK>': 1, '<s>': 2, '</s>': 3}
    for vo in tgt_vocab:
        if tw2i.get(vo) is None:
            tw2i[vo] = len(tw2i)
    ti2w = {v: k for k, v in tw2i.items()}

    X_p, Y_p = [], []
    for s, t in zip(X_r, Y_r):
        X_p.append(prepare_sequence(s + ['</s>'], sw2i).view(1, -1))
        Y_p.append(prepare_sequence(t + ['</s>'], tw2i).view(1, -1))

    train_data = list(zip(X_p, Y_p))
    return train_data, sw2i, si2w, tw2i, ti2w
