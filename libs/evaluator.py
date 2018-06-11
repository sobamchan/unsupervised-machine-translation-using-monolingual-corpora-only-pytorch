from libs import utils


class Evaluator:

    def __init__(self, trainer):
        self.args = trainer.args
        self.trainer = trainer
        self.logger = trainer.logger
        self.test_loader = trainer.test_loader
        self.src_vocab = trainer.src_vocab
        self.tgt_vocab = trainer.tgt_vocab

    def bleu(self, log_dict):
        for dict_ in self.test_loader:
            src_sents, tgt_sents = dict_['src'], dict_['tgt']
            pred_sents = self.trainer.translate_batch(src_sents,
                                                      tgt_sents[:])
            log_dict['test_bleus'] +=\
                [utils.calc_bleu(t, p) for t, p in zip(tgt_sents, pred_sents)]

    def sample_translation(self, sample_n=3):
        src_sents = self.test_loader.dataset.src_sents[:3]
        tgt_sents = self.test_loader.dataset.tgt_sents[:3]

        log = []
        for s, t in zip(src_sents, tgt_sents):
            _log = {}
            _log['src'] = s
            _log['tgt'] = t
            _log['pred'] = self.trainer.translate([s], 'src')[0]
            log.append(_log)
        return log
