import os


class SmallParallelEnJa:

    def __init__(self, args, train=True):
        self.args = args
        self.data_dir = args.data_dir
        self.train = train

    def load_dataset(self):
        if self.train:
            src_path = os.path.join(self.data_dir, 'train.en')
            tgt_path = os.path.join(self.data_dir, 'train.ja')
        else:
            src_path = os.path.join(self.data_dir, 'test.en')
            tgt_path = os.path.join(self.data_dir, 'test.ja')

        with open(src_path, 'r') as f:
            src_sents = [line.strip() for line in f.readlines()]
        with open(tgt_path, 'r') as f:
            tgt_sents = [line.strip() for line in f.readlines()]

        self.src_sents = src_sents
        self.tgt_sents = tgt_sents

        return src_sents, tgt_sents
