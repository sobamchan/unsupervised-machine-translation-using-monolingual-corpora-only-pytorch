from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction


def pad_to_batch(src_sents, tgt_sents, src_pad_idx, tgt_pad_idx):
    '''
    in:
      - src/tgt_sents: [[word_id, ...], ...]
    '''
    batch = list(zip(src_sents, tgt_sents))
    sorted_batch = sorted(batch,
                          key=lambda b: len(b[0]),
                          reverse=True)
    src_sents, tgt_sents = list(zip(*sorted_batch))
    src_max_len = max([len(sent) for sent in src_sents])
    tgt_max_len = max([len(sent) for sent in tgt_sents])
    src_p, tgt_p = [], []
    src_lens, tgt_lens = [], []

    for i in range(len(batch)):
        src_lens.append(len(src_sents[i]))
        if len(src_sents[i]) < src_max_len:
            pads = [src_pad_idx] * (src_max_len - len(src_sents[i]))
            src_p.append(src_sents[i] + pads)
        else:
            src_p.append(src_sents[i])

        tgt_lens.append(len(tgt_sents[i]))
        if len(tgt_sents[i]) < tgt_max_len:
            pads = [tgt_pad_idx] * (tgt_max_len - len(tgt_sents[i]))
            tgt_p.append(tgt_sents[i] + pads)
        else:
            tgt_p.append(tgt_sents[i])

    return src_p, src_lens, tgt_p, tgt_lens


def calc_bleu(ref, pred):
    sf = SmoothingFunction().method4
    ref = [ref]
    return sentence_bleu(ref, pred, smoothing_function=sf)
