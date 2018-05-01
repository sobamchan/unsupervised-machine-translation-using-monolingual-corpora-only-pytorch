from pathlib import Path
from torch.utils import data
from torch.utils.data.sampler import RandomSampler
from collections import Counter
from libs import utils


def get_dataset(src_path, tgt_path):
    src_sents = open(src_path, 'r', encoding='utf-8').readlines()
    tgt_sents = open(tgt_path, 'r', encoding='utf-8').readlines()
    if src_path.find('ja') == -1:
        x = [utils.normalize_string(s).split() for s in src_sents]
    else:
        x = [s.lower().split() for s in src_sents]
    if tgt_path.find('ja') == -1:
        y = [utils.normalize_string(s).split() for s in tgt_sents]
    else:
        y = [s.lower().split() for s in tgt_sents]
    return x, y


def build_vocab(words, vocab_size):
    if vocab_size:
        vocab = Counter(list(set(words)))
        vocab = [cnt[0] for cnt in vocab.most_common(vocab_size)]
    else:
        vocab = list(set(words))
    w2i = {'<PAD>': 0, '<UNK>': 1, '<s>': 2, '</s>': 3}
    for vo in vocab:
        if w2i.get(vo) is None:
            w2i[vo] = len(w2i)
    i2w = {v: k for k, v in w2i.items()}
    return vocab, w2i, i2w


class Dataset(data.Dataset):

    def __init__(self, src_path, tgt_path, test=False,
                 src_vocab_size=None, tgt_vocab_size=None,
                 sw2i=None, tw2i=None):
        src, tgt = get_dataset(src_path, tgt_path)

        if test:
            svocab = list(sw2i.keys())
            tvocab = list(tw2i.keys())
            si2w = {v: k for k, v in sw2i.items()}
            ti2w = {v: k for k, v in tw2i.items()}
        else:
            svocab, sw2i, si2w = build_vocab(utils.flatten(src),
                                             src_vocab_size)
            tvocab, tw2i, ti2w = build_vocab(utils.flatten(tgt),
                                             tgt_vocab_size)

        # src_p, tgt_p = [], []
        # for s, t in zip(src, tgt):
        #     src_p.append(utils.prepare_sequence(s + ['</s>'],
        #                                         sw2i).view(1, -1))
        #     tgt_p.append(utils.prepare_sequence(t + ['</s>'],
        #                                         tw2i).view(1, -1))

        self.svocab = svocab
        self.tvocab = tvocab
        self.sw2i = sw2i
        self.tw2i = tw2i
        self.si2w = si2w
        self.ti2w = ti2w
        # self.src = src_p
        # self.tgt = tgt_p
        self.src = [' '.join(s) for s in src]
        self.tgt = [' '.join(t) for t in tgt]

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return {'src': self.src[idx], 'tgt': self.tgt[idx]}


def get_dataloaders(data_dir, src_lang, tgt_lang, batch_size,
                    src_vocab_size, tgt_vocab_size):
    data_dir = Path(data_dir)
    src_path = data_dir / ('train.%s' % src_lang)
    tgt_path = data_dir / ('train.%s' % tgt_lang)
    train_dataset = Dataset(str(src_path),
                            str(tgt_path),
                            test=False,
                            src_vocab_size=src_vocab_size,
                            tgt_vocab_size=tgt_vocab_size)
    src_path = data_dir / ('test.%s' % src_lang)
    tgt_path = data_dir / ('test.%s' % tgt_lang)
    test_dataset = Dataset(str(src_path),
                           str(tgt_path),
                           test=True,
                           sw2i=train_dataset.sw2i,
                           tw2i=train_dataset.tw2i)

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   sampler=RandomSampler(train_dataset))
    test_loader = data.DataLoader(test_dataset,
                                  batch_size=batch_size,
                                  shuffle=False)
    return train_loader, test_loader
