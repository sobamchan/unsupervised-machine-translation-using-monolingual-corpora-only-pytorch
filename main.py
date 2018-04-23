import os
import argparse
from distutils.util import strtobool
import numpy as np
import torch
from libs.trainer import Trainer
from libs.evaluator import Evaluator
from libs.logger import Logger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',
                        type=str,
                        default='../DATA/giga-fren')
    parser.add_argument('--output-dir',
                        type=str,
                        default='./test')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--src-lang', type=str, default='en')
    parser.add_argument('--tgt-lang', type=str, default='fr')
    parser.add_argument('--src-vocab-size', type=int, default=25000)
    parser.add_argument('--tgt-vocab-size', type=int, default=25000)
    parser.add_argument('--encoder-hidden-n', type=int, default=256)
    parser.add_argument('--encoder-layers-n', type=int, default=3)
    parser.add_argument('--encoder-embedding-dim', type=int, default=256)
    parser.add_argument('--encoder-bidirec', type=strtobool, default='1')
    parser.add_argument('--decoder-hidden-n', type=int, default=256)
    parser.add_argument('--decoder-layers-n', type=int, default=3)
    parser.add_argument('--decoder-embedding-dim', type=int, default=256)
    parser.add_argument('--dropout-p', type=float, default=0.1)
    parser.add_argument('--use-cuda', type=strtobool, default='1')
    parser.add_argument('--gpu-id', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def main(args):
    logger = Logger(args.output_dir)
    args.logger = logger
    trainer = Trainer(args)
    evaluator = Evaluator(trainer)
    for i_epoch in range(0, args.epoch + 1):

        # train
        log_dict = {'i_epoch': i_epoch,
                    'train_losses': [],  # per batch
                    'test_bleus': []}   # per sample
        trainer.train_one_epoch(log_dict)

        # evaluation and logging
        logger.log('%d th epoch' % i_epoch)
        evaluator.bleu(log_dict)
        evaluator.sample_translation()
        log_dict_mean = {'i_epoch': log_dict['i_epoch'],
                         'train_loss': np.mean(log_dict['train_losses']),
                         'test_bleu': np.mean(log_dict['test_bleus'])}
        logger.dump(log_dict_mean)
        trainer.save_best(log_dict_mean)
        logger.log('-' * 10)


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
