from libs import word_translation
from libs.vocabulary import Vocabulary
import numpy as np


if __name__ == '__main__':

    envec = np.load('input/vectors-en.npy')
    frvec = np.load('input/vectors-fr.npy')
    with open('input/vectors-en-vocab.txt') as f:
        enwords = [line.strip() for line in f.readlines()]
    with open('input/vectors-fr-vocab.txt') as f:
        frwords = [line.strip() for line in f.readlines()]
    envocab = Vocabulary(75000)
    envocab.build_vocab_from_words(enwords)
    frvocab = Vocabulary(75000)
    frvocab.build_vocab_from_words(frwords)

    WT = word_translation.WordTranslator(envec, envocab, frvec, frvocab)
    # sent = input('type sentence to translate: ')
    sent = 'dog'
    result = WT.translate(sent)
    print('%s -> %s' % (sent, result))
