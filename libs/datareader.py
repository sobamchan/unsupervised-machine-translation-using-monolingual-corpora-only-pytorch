from pathlib import Path


class DataReader:

    def __init__(self, args, train=True):
        self.args = args
        self.data_dir = Path(args.data_dir)
        self.train = train

    def load_dataset(self):
        args = self.args
        if self.train:
            src_path = self.data_dir / ('train.%s' % args.src_lang)
            tgt_path = self.data_dir / ('train.%s' % args.tgt_lang)
        else:
            src_path = self.data_dir / ('test.%s' % args.src_lang)
            tgt_path = self.data_dir / ('test.%s' % args.tgt_lang)

        with open(src_path, 'r') as f:
            src_sents = [line.strip() for line in f.readlines()]
        with open(tgt_path, 'r') as f:
            tgt_sents = [line.strip() for line in f.readlines()]
        return src_sents, tgt_sents
