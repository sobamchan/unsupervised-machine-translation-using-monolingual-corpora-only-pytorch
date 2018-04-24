import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from libs.dataset import get_dataloaders
from libs import utils
from libs import models


class Trainer:

    def __init__(self, args):
        self.args = args
        train_dataloader, test_dataloader =\
            get_dataloaders(args.data_dir,
                            args.src_lang,
                            args.tgt_lang,
                            args.batch_size,
                            args.src_vocab_size,
                            args.tgt_vocab_size)

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.src_vocab = self.train_dataloader.dataset.src_vocab
        self.src_w2i = self.train_dataloader.dataset.src_w2i
        self.src_i2w = self.train_dataloader.dataset.src_i2w
        self.tgt_vocab = self.train_dataloader.dataset.tgt_vocab
        self.tgt_w2i = self.train_dataloader.dataset.tgt_w2i
        self.tgt_i2w = self.train_dataloader.dataset.tgt_i2w
        vocab_objs = {
                'tgt': {
                    'vocab': self.tgt_vocab,
                    'w2i': self.tgt_w2i,
                    'i2w': self.tgt_i2w
                    },
                'src': {
                    'vocab': self.src_vocab,
                    'w2i': self.src_w2i,
                    'i2w': self.src_i2w
                    }
                }
        self.vocab_objs = vocab_objs

        # model
        encoder = models.Encoder(len(self.src_vocab),
                                 args.src_embedding_size,
                                 self.src_w2i['<PAD>'],
                                 args.encoder_dropout_p,
                                 args.encoder_hidden_n,
                                 args.encoder_num_layers,
                                 args.encoder_bidirectional,
                                 args.use_cuda)
        decoder = models.Decoder(len(self.tgt_vocab),
                                 args.tgt_embedding_size,
                                 self.tgt_w2i['<PAD>'],
                                 args.decoder_dropout_p,
                                 args.decoder_hidden_n,
                                 args.decoder_num_layers,
                                 args.decoder_bidirectional,
                                 args.use_cuda)
        src_embedder = models.Embedder(len(self.src_vocab),
                                       args.src_embedding_size,
                                       self.src_w2i['<PAD>'])
        tgt_embedder = models.Embedder(len(self.tgt_vocab),
                                       args.tgt_embedding_size,
                                       self.tgt_w2i['<PAD>'])
        if args.use_cuda:
            encoder.cuda()
            decoder.cuda()
            src_embedder.cuda()
            tgt_embedder.cuda()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedder = src_embedder
        self.tgt_embedder = tgt_embedder

        # optimizer
        self.enc_optim = optim.SGD(self.encoder.parameters(), args.lr)
        self.dec_optim = optim.SGD(self.decoder.parameters(), args.lr)
        self.src_emb_optim = optim.SGD(self.src_embedder.parameters(), args.lr)
        self.tgt_emb_optim = optim.SGD(self.tgt_embedder.parameters(), args.lr)

    def train_one_epoch(self, log_dict):
        args = self.args
        src_w2i = self.src_w2i
        tgt_w2i = self.tgt_w2i
        self.encoder.train()
        self.decoder.train()
        self.src_embedder.train()
        self.tgt_embedder.train()

        losses = []
        for batch in tqdm(self.train_dataloader):

            self.encoder.zero_grad()
            self.decoder.zero_grad()
            self.src_embedder.zero_grad()
            self.tgt_embedder.zero_grad()

            src_sents = batch['src']
            tgt_sents = batch['tgt']
            src_sents = [utils.convert_s2i(sent, src_w2i)
                         for sent in src_sents]
            tgt_sents = [utils.convert_s2i(sent, tgt_w2i)
                         for sent in tgt_sents]
            src_sents, tgt_sents, src_lens, tgt_lens =\
                utils.pad_batch(src_sents,
                                tgt_sents,
                                src_w2i['<PAD>'],
                                tgt_w2i['<PAD>'])

            src_sents = Variable(torch.LongTensor(src_sents))
            tgt_sents = Variable(torch.LongTensor(tgt_sents))
            if args.use_cuda:
                src_sents = src_sents.cuda()
                tgt_sents = tgt_sents.cuda()

            outputs, hidden = self.encoder(self.src_embedder,
                                           src_sents,
                                           src_lens)
            preds = self.decoder(self.tgt_embedder,
                                 self.tgt_w2i['<s>'],
                                 outputs,
                                 hidden,
                                 tgt_sents.size(1))
            loss = F.cross_entropy(preds, tgt_sents.view(-1))
            loss.backward()
            self.enc_optim.step()
            self.dec_optim.step()
            self.src_emb_optim.step()
            self.tgt_emb_optim.step()
            log_dict['train_losses'].append(loss.data[0])
            losses.append(loss.data[0])
        print(np.mean(losses))

    def translation_validate(self):
        args = self.args
        src_w2i = self.src_w2i
        tgt_w2i = self.tgt_w2i
        self.encoder.eval()
        self.decoder.eval()

        losses = []
        accs = []
        for batch in tqdm(self.test_dataloader):

            src_sents = batch['src']
            tgt_sents = batch['tgt']
            src_sents = [utils.convert_s2i(sent, src_w2i)
                         for sent in src_sents]
            tgt_sents = [utils.convert_s2i(sent, tgt_w2i)
                         for sent in tgt_sents]
            src_sents, tgt_sents, src_lens, tgt_lens =\
                utils.pad_batch(src_sents,
                                tgt_sents,
                                src_w2i['<PAD>'],
                                tgt_w2i['<PAD>'])

            src_sents = Variable(torch.LongTensor(src_sents), volatile=True)
            tgt_sents = Variable(torch.LongTensor(tgt_sents), volatile=True)
            if args.use_cuda:
                src_sents = src_sents.cuda()
                tgt_sents = tgt_sents.cuda()

            outputs, hidden = self.encoder(src_sents, src_lens)
            preds = self.decoder(self.tgt_w2i['<s>'],
                                 outputs,
                                 hidden,
                                 tgt_sents.size(1))

            loss = F.cross_entropy(preds, tgt_sents.view(-1))
            losses.append(loss.data[0])

            batch_size = len(batch['src'])
            preds = preds.view(batch_size, max(tgt_lens), -1)
            preds = preds.max(2)[1]

            for tgt_sent, tgt_len, pred in zip(tgt_sents.data,
                                               tgt_lens,
                                               preds.data):
                pred = np.array(pred[:tgt_len])
                tgt_sent = np.array(tgt_sent[:tgt_len])
                acc = np.equal(pred, tgt_sent).mean()
                accs.append(acc)

        print('sample translation')
        print('source: %s' % batch['src'][0])
        print('target: %s' % batch['tgt'][0])
        tgt_i2w = self.tgt_i2w
        print('prediction: %s' % [tgt_i2w[i.data[0]] for i in preds[0]])
        print('test loss %f' % np.mean(losses))
        print('test accuracy %f' % np.mean(accs))

    def autoencoder(self, subj):
        '''
        in:
          subj: string src or tgt
        '''
        args = self.args
        src_w2i = self.src_w2i
        tgt_w2i = self.tgt_w2i

        losses = []
        for batch in tqdm(self.train_dataloader):

            self.encoder.zero_grad()
            self.decoder.zero_grad()

            src_sents = batch['src']
            tgt_sents = batch['tgt']
            src_sents = [utils.convert_s2i(sent, src_w2i)
                         for sent in src_sents]
            tgt_sents = [utils.convert_s2i(sent, tgt_w2i)
                         for sent in tgt_sents]
            src_sents, tgt_sents, src_lens, tgt_lens =\
                utils.pad_batch(src_sents,
                                tgt_sents,
                                src_w2i['<PAD>'],
                                tgt_w2i['<PAD>'])

            src_sents = Variable(torch.LongTensor(src_sents))
            tgt_sents = Variable(torch.LongTensor(tgt_sents))
            if args.use_cuda:
                src_sents = src_sents.cuda()
                tgt_sents = tgt_sents.cuda()

            outputs, hidden = self.encoder(src_sents, src_lens)
            preds = self.decoder(self.tgt_w2i['<s>'],
                                 outputs,
                                 hidden,
                                 tgt_sents.size(1))
            loss = F.cross_entropy(preds, tgt_sents.view(-1))
            loss.backward()
            self.enc_optim.step()
            self.dec_optim.step()
            losses.append(loss.data[0])
        print(np.mean(losses))
