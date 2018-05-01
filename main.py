import os
import argparse
from distutils.util import strtobool
import torch
from libs.trainer import Trainer
# from libs.evaluator import Evaluator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--data-dir',
                        type=str,
                        default='../DATA/small-europarl-v7')
    parser.add_argument('--output-dir',
                        type=str,
                        default='./test')
    parser.add_argument('--src-lang', type=str, default='fr')
    parser.add_argument('--tgt-lang', type=str, default='en')

    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)

    parser.add_argument('--src-vocab-size', type=int, default=30000)
    parser.add_argument('--tgt-vocab-size', type=int, default=30000)
    parser.add_argument('--src-embedding-size', type=int, default=512)
    parser.add_argument('--encoder-dropout-p', type=float, default=0.1)
    parser.add_argument('--encoder-hidden-n', type=int, default=512)
    parser.add_argument('--encoder-num-layers', type=int, default=1)
    parser.add_argument('--tgt-embedding-size', type=int, default=512)
    parser.add_argument('--decoder-dropout-p', type=float, default=0.1)
    parser.add_argument('--decoder-hidden-n', type=int, default=512)
    parser.add_argument('--decoder-num-layers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--use-cuda', type=strtobool, default='1')
    parser.add_argument('--encoder-bidirectional', type=strtobool, default='0')
    parser.add_argument('--decoder-bidirectional', type=strtobool, default='0')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def main(args):
    print(args)
    trainer = Trainer(args)
    # evaluator = Evaluator(trainer)
    for i_epoch in range(0, args.epoch + 1):

        # train
        # log_dict = {'i_epoch': i_epoch,
        #             'train_losses': [],  # per batch
        #             'test_bleus': []}   # per sample
        # trainer.train_one_epoch_translator()
        trainer.train_one_epoch_autoencoder('tgt')

        # trainer.translation_validate()

        # evaluation and logging
        # evaluator.bleu(log_dict)
        # evaluator.sample_translation()


if __name__ == '__main__':

    args = get_args()

    # GPU usage
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    print('using GPU id: ', os.environ['CUDA_VISIBLE_DEVICES'])
    if args.use_cuda and torch.cuda.is_available():
        args.use_cuda = True
    else:
        args.use_cuda = False

    main(args)
