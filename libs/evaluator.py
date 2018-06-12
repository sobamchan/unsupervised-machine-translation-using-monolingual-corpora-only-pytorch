

class Evaluator:

    def __init__(self, trainer):
        self.args = trainer.args
        self.trainer = trainer
        # self.logger = trainer.logger
        self.test_loader = trainer.test_dataloader

    def sample_translation(self, sample_n=3):
        src_sents = self.test_loader.dataset.src[:3]
        tgt_sents = self.test_loader.dataset.tgt[:3]

        log = []
        for s, t in zip(src_sents, tgt_sents):
            _log = {}
            _log['src'] = s
            _log['tgt'] = t
            _log['pred'] = self.trainer.translate([s], 'src')[0]
            log.append(_log)
        return log
