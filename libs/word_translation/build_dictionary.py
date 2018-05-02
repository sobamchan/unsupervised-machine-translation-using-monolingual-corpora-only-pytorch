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


def main():
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


if __name__ == '__main__':
    main()
