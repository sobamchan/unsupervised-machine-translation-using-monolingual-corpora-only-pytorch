import argparse
import json
import numpy as np
from gensim.models import KeyedVectors
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang1-path',
                        type=str,
                        default='./input/vectors-en.txt')
    parser.add_argument('--lang2-path',
                        type=str,
                        default='./input/vectors-fr.txt')
    parser.add_argument('--output-path',
                        type=str)
    return parser.parse_args()


def from_gensim_txt_file():
    args = get_args()
    lang1_vec = KeyedVectors.load_word2vec_format(args.lang1_path,
                                                  binary=False)
    lang2_vec = KeyedVectors.load_word2vec_format(args.lang2_path,
                                                  binary=False)

    dict1to2 = {}
    for lang1_word in tqdm(lang1_vec.index2word):
        vec = lang1_vec.get_vector(lang1_word)
        idx = np.argmax(np.dot(lang2_vec.vectors, vec)).argmax()
        dict1to2[lang1_word] = lang2_vec.index2word[idx]

    with open(args.output_path, 'w') as f:
        json.dump(dict1to2, f, ensure_ascii=False, indent=2)


def from_npy_file():
    # args = get_args()
    # vecs1 = np.load(args.lang1_path)
    # vecs2 = np.load(args.lang2_path)

    vecs1 = np.load('./input/vectors-en.npy')
    vecs2 = np.load('./input/vectors-fr.npy')

    b = 256
    idxs = []
    for i in range(0, len(vecs1), b):
        scores = np.dot(vecs1[i:i+256], vecs2.transpose())  # b, len(vecs2)
        part_idxs = np.argmax(scores, 1)  # b
        idxs.append(part_idxs)

    return idxs
