import json


class Dictionary:

    def __init__(self, fpath):
        self.fpath = fpath
        with open(fpath, 'r') as f:
            d = json.load(f)
        self.d = d

    def translate(self, words, src_to_tgt=True):
        if src_to_tgt:
            d = self.d
        else:
            d = {v: k for k, v in self.d.items()}
        result_words = []
        for w in words:
            if w in d.keys():
                result_words.append(d[w])
            else:
                result_words.append('<UNK>')
        return result_words
