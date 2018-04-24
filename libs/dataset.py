from pathlib import Path
from collections import Counter
from torch.utils import data
from torch.utils.data.sampler import RandomSampler


def read_datset(src_path, tgt_path):
    with open(src_path, 'r') as f:
        src_lines = [line.strip() for line in f.readlines()]
    with open(tgt_path, 'r') as f:
        tgt_lines = [line.strip() for line in f.readlines()]
    return src_lines, tgt_lines


def build_vocab(lines, vocab_size=75000):
    counter = Counter()
    for line in lines:
        for word in line.split():
            counter[word] += 1
    vocab = [v[0] for v in counter.most_common(vocab_size)]
    w2i = {'<PAD>': 0, '<UNK>': 1, '<s>': 2, '</s>': 3}
    for v in vocab:
        if v not in w2i.keys():
            w2i[v] = len(w2i)
    i2w = {v: k for k, v in w2i.items()}
    for token in ['<PAD>', '<UNK>', '<s>', '</s>']:
        if token not in vocab:
            vocab.append(token)
    return vocab, w2i, i2w


class Dataset(data.Dataset):

    def __init__(self, src_path, tgt_path,
                 src_vocab_size=None,
                 tgt_vocab_size=None):
        src_lines, tgt_lines = read_datset(src_path, tgt_path)
        self.src_lines = src_lines
        self.tgt_lines = tgt_lines
        src_vocab, src_w2i, src_i2w = build_vocab(src_lines, src_vocab_size)
        self.src_vocab = src_vocab
        self.src_w2i = src_w2i
        self.src_i2w = src_i2w
        tgt_vocab, tgt_w2i, tgt_i2w = build_vocab(tgt_lines, tgt_vocab_size)
        self.tgt_vocab = tgt_vocab
        self.tgt_w2i = tgt_w2i
        self.tgt_i2w = tgt_i2w

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        return {'src': self.src_lines[idx], 'tgt': self.tgt_lines[idx]}


def get_dataloaders(data_dir, src_lang, tgt_lang, batch_size,
                    src_vocab_size=None, tgt_vocab_size=None):
    data_dir = Path(data_dir)
    src_path = data_dir / ('train.%s' % src_lang)
    tgt_path = data_dir / ('train.%s' % tgt_lang)
    train_dataset = Dataset(str(src_path),
                            str(tgt_path),
                            src_vocab_size,
                            tgt_vocab_size)
    src_path = data_dir / ('test.%s' % src_lang)
    tgt_path = data_dir / ('test.%s' % tgt_lang)
    test_dataset = Dataset(str(src_path),
                           str(tgt_path),
                           src_vocab_size,
                           tgt_vocab_size)

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   sampler=RandomSampler(train_dataset))
    test_loader = data.DataLoader(test_dataset,
                                  batch_size=batch_size,
                                  shuffle=False)
    return train_loader, test_loader
