import numpy as np


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
