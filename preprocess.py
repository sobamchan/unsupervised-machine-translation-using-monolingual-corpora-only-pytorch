import os
from pathlib import Path
import numpy as np
import pickle
from gensim.models import KeyedVectors
from libs.vocabulary import Vocabulary


def build_vocabo_and_vec_npy(vec_file_path, vocab_size, odir=None):
    '''
    From raw fastText vector txt file
    TO
    libs.vocabulary.Vocabulary object and vectors in np object
    '''
    w2v = KeyedVectors.load_word2vec_format(vec_file_path,
                                            binary=False)

    if not vocab_size:
        vocab_size = len(w2v.index2word)

    words = w2v.index2word[:vocab_size]
    vocab = Vocabulary(False, False, words)
    vectors = w2v.vectors[:vocab_size]

    new_random_vectors_n = len(vocab) - len(vectors)
    new_random_vectors = np.random.rand(new_random_vectors_n, vectors.shape[1])
    vectors = np.concatenate((new_random_vectors, vectors))

    if odir:
        original_vectors_fname =\
            os.path.splitext(os.path.basename(vec_file_path))[0]
        vectors_fname = '%s-%d.npy' % (original_vectors_fname, vocab_size)
        vectors_path = os.path.join(odir, vectors_fname)
        np.save(vectors_path, vectors)

        vocab_fname = '%s-%d.vocab.pth' % (original_vectors_fname, vocab_size)
        vocab_path = os.path.join(odir, vocab_fname)
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f)

    return vocab, vectors


def vector_txt_file_to_npy_and_words_list(fpath):
    fpath = Path(fpath)
    w2v = KeyedVectors.load_word2vec_format(str(fpath),
                                            binary=False)
    words = list(w2v.index2word)
    vectors = w2v.vectors

    np.save(fpath.parent / (fpath.stem + '.npy'), vectors)
    with open(fpath.parent / (fpath.stem + '-vocab.txt'), 'w') as f:
        f.write('\n'.join(words))
