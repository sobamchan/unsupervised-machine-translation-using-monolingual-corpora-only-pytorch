import io
import numpy as np
from tqdm import tqdm


# https://github.com/facebookresearch/MUSE/blob/master/demo.ipynb
def load_vec(emb_path, nmax=50000):
    vectors = []
    w2i = {}
    with io.open(emb_path, 'r', encoding='utf-8',
                 newline='\n', errors='\n') as f:
        next(f)
        for i, line in enumerate(f):
            word, vec = line.rstrip().split(' ', 1)
            vec = np.fromstring(vec, sep=' ')
            assert word not in w2i, 'word found twice'
            vectors.append(vec)
            w2i[word] = len(w2i)
            if len(w2i) == nmax:
                break
    i2w = {v: k for k, v in w2i.items()}
    embeddings = np.vstack(vectors)
    return embeddings, i2w, w2i


def get_nn(word, src_emb, si2w, tgt_emb, ti2w, K=5):
    w2i = {v: k for k, v in si2w.items()}
    word_emb = src_emb[w2i[word]]
    scores = (tgt_emb / np.linalg.norm(tgt_emb, 2, 1)[:, None])\
        .dot(word_emb / np.linalg.norm(word_emb))
    k_best = scores.argsort()[-K:][::-1]
    for i, idx in enumerate(k_best):
        print('%.4f - %s' % (scores[idx], ti2w[idx]))


def translate(words, sembs, si2w, tembs, ti2w):
    translated_words = []
    w2i = {v: k for k, v in si2w.items()}
    for word in words:
        if word in w2i.keys():
            word_emb = sembs[w2i[word]]
            scores = (tembs / np.linalg.norm(tembs, 2, 1)[:, None])\
                .dot(word_emb / np.linalg.norm(word_emb))
            pred = scores.argsort()[-1]
            translated_words.append(ti2w[pred])
        else:
            translated_words.append('<UNK>')
    return translated_words


def translate_file(src_file, swap=False):
    s_embs, si2w, sw2i, t_embs, ti2w, tw2i = main(swap)
    s_lines = open(src_file).readlines()
    pred_lines = []
    for s_line in tqdm(s_lines):
        s_words = s_line.lower().split()
        pred_words = translate(s_words, s_embs, si2w, t_embs, ti2w)
        pred_lines.append(' '.join(pred_words))
    return pred_lines


def main(swap=False):
    if not swap:
        src_path = './input/vectors-en.txt'
        tgt_path = './input/vectors-fr.txt'
    else:
        tgt_path = './input/vectors-en.txt'
        src_path = './input/vectors-fr.txt'
    nmax = 50000
    s_embs, si2w, sw2i = load_vec(src_path, nmax)
    t_embs, ti2w, tw2i = load_vec(tgt_path, nmax)
    return s_embs, si2w, sw2i, t_embs, ti2w, tw2i
