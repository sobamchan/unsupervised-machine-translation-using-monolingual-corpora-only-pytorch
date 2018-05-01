import argparse
import numpy as np
from tqdm import tqdm
from libs.vocabulary import Vocabulary


class WordTranslator:

    def __init__(self, src_vecs, src_vocab, tgt_vecs, tgt_vocab):
        self.src_vecs = src_vecs
        self.src_vocab = src_vocab
        self.tgt_vecs = tgt_vecs
        self.tgt_vocab = tgt_vocab

    def translate(self, sent):
        '''
        in:
          word_idxes: S
        '''
        src_vecs = self.src_vecs  # V, D
        tgt_vecs = self.tgt_vecs  # V, D
        # normalize
        src_vecs = src_vecs / np.linalg.norm(src_vecs, 2, 1, keepdims=True)
        tgt_vecs = tgt_vecs / np.linalg.norm(tgt_vecs, 2, 1, keepdims=True)

        word_idxes = self.src_vocab.encode(sent)[0]
        word_vecs = src_vecs[word_idxes]  # S, D
        scores = np.dot(word_vecs, tgt_vecs.transpose())  # S, V

        selected_word_idxes = np.argmax(scores, axis=1)
        return self.tgt_vocab.decode(selected_word_idxes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file',
                        type=str,
                        required=True)
    parser.add_argument('--output-file',
                        type=str,
                        required=True)
    parser.add_argument('--src-vec-file',
                        type=str,
                        default='input/vectors-en.npy',
                        help='npy file')
    parser.add_argument('--tgt-vec-file',
                        type=str,
                        default='input/vectors-fr.npy',
                        help='npy file')
    parser.add_argument('--src-vocab-file',
                        type=str,
                        default='input/vectors-en-vocab.txt',
                        help='txt file')
    parser.add_argument('--tgt-vocab-file',
                        type=str,
                        default='input/vectors-fr-vocab.txt',
                        help='txt file')
    parser.add_argument('--src-vocab-size',
                        type=int,
                        default=int(1e+10))
    parser.add_argument('--tgt-vocab-size',
                        type=int,
                        default=int(1e+10))
    args = parser.parse_args()

    envec = np.load(args.src_vec_file)
    frvec = np.load(args.tgt_vec_file)
    with open(args.src_vocab_file) as f:
        enwords = [line.strip() for line in f.readlines()]
    with open(args.tgt_vocab_file) as f:
        frwords = [line.strip() for line in f.readlines()]
    envocab = Vocabulary(args.src_vocab_size)
    envocab.build_vocab_from_words(enwords)
    frvocab = Vocabulary(args.tgt_vocab_size)
    frvocab.build_vocab_from_words(frwords)

    WT = WordTranslator(envec, envocab, frvec, frvocab)
    input_lines = open(args.input_file).readlines()
    output_f = open(args.output_file, 'a')
    for input_line in tqdm(input_lines):
        output_line = WT.translate(input_line)
        output_f.write(output_line)
        output_f.write('\n')
