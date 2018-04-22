import torch.utils.data as data
from torch.utils.data.sampler import RandomSampler

from libs.datareader import DataReader
from libs.vocabulary import Vocabulary


class Dataset(data.DataLoader):

    def __init__(self, src_sents, tgt_sents, args,
                 src_vocab, tgt_vocab, train=True):

        self.src_sents = src_sents
        self.tgt_sents = tgt_sents

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.src_sents)

    def __getitem__(self, idx):
        src = self.src_sents[idx]
        src = src + ' ' + '</s>'

        tgt = self.tgt_sents[idx]
        tgt = tgt + ' ' + '</s>'

        dict_ = {'src': src,
                 'tgt': tgt}
        return dict_


def get_dataloader(args):
    src_sents, tgt_sents = DataReader(args, train=True).load_dataset()

    # with open('input/vectors-%s-vocab.txt' % args.src_lang) as f:
    #     src_words = [line.strip() for line in f.readlines()]
    # with open('input/vectors-%s-vocab.txt' % args.tgt_lang) as f:
    #     tgt_words = [line.strip() for line in f.readlines()]
    src_vocab = Vocabulary(args.src_vocab_size)
    tgt_vocab = Vocabulary(args.tgt_vocab_size)
    # src_vocab.build_vocab_from_words(src_words)
    # tgt_vocab.build_vocab_from_words(tgt_words)
    src_vocab.build_vocab_from_sents(src_sents)
    tgt_vocab.build_vocab_from_sents(tgt_sents)

    train_dataset = Dataset(src_sents,
                            tgt_sents,
                            args,
                            src_vocab,
                            tgt_vocab,
                            train=True)

    src_sents, tgt_sents = DataReader(args, train=False).load_dataset()
    test_dataset = Dataset(src_sents,
                           tgt_sents,
                           args,
                           src_vocab,
                           tgt_vocab,
                           train=False)

    train_dataloader = data.DataLoader(train_dataset,
                                       args.batch_size,
                                       sampler=RandomSampler(train_dataset))
    test_dataloader = data.DataLoader(test_dataset,
                                      args.batch_size,
                                      shuffle=False)

    return train_dataloader, test_dataloader
