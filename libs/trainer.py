import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import LongTensor as LT
from torch.autograd import Variable
from libs import models
from libs import utils
from libs.dataset import get_dataloaders
from tqdm import tqdm


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
        self.sw2i = train_dataloader.dataset.sw2i
        self.si2w = train_dataloader.dataset.si2w
        self.tw2i = train_dataloader.dataset.tw2i
        self.ti2w = train_dataloader.dataset.ti2w
        self.converters = {
                'src': {
                    'w2i': self.sw2i,
                    'i2w': self.si2w
                    },
                'tgt': {
                    'w2i': self.tw2i,
                    'i2w': self.ti2w
                    }
                }

        vocab_size = max(len(self.sw2i), len(self.tw2i))
        print('global vocab size: %d' % vocab_size)

        encoder = models.Encoder(vocab_size,
                                 args.src_embedding_size,
                                 args.encoder_hidden_n,
                                 n_layers=args.encoder_num_layers,
                                 bidirec=args.encoder_bidirectional,
                                 use_cuda=args.use_cuda)
        if args.decoder_bidirectional:
            decoder_hidden_size = args.decoder_hidden_n * 2
        else:
            decoder_hidden_size = args.decoder_hidden_n
        decoder = models.Decoder(vocab_size,
                                 args.tgt_embedding_size,
                                 decoder_hidden_size,
                                 n_layers=args.decoder_num_layers,
                                 use_cuda=args.use_cuda)
        src_embedder = models.Embedder(vocab_size,
                                       args.src_embedding_size,
                                       args.use_cuda)
        tgt_embedder = models.Embedder(vocab_size,
                                       args.tgt_embedding_size,
                                       args.use_cuda)
        encoder.init_weight()
        decoder.init_weight()
        if args.use_cuda:
            encoder = encoder.cuda()
            decoder = decoder.cuda()
            src_embedder = src_embedder.cuda()
            tgt_embedder = tgt_embedder.cuda()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedder = src_embedder
        self.tgt_embedder = tgt_embedder
        self.embedders = {
                'src': src_embedder,
                'tgt': tgt_embedder
                }
        self.loss_func = nn.CrossEntropyLoss(ignore_index=0)
        self.enc_optim = optim.Adam(encoder.parameters(), lr=args.lr)
        self.dec_optim = optim.Adam(decoder.parameters(), lr=args.lr)
        self.src_embedder_optim = optim.Adam(self.src_embedder.parameters(),
                                             lr=args.lr)
        self.tgt_embedder_optim = optim.Adam(self.tgt_embedder.parameters(),
                                             lr=args.lr)
        self.optims = {
                'src': self.src_embedder_optim,
                'tgt': self.tgt_embedder_optim,
                }

    def train_one_epoch_translator(self, _from='src', _to='tgt'):
        print('%s -> %s' % (_from, _to))
        sw2i = self.converters[_from]['w2i']
        tw2i = self.converters[_to]['w2i']
        ti2w = self.converters[_to]['i2w']
        src_embedder = self.embedders[_from]
        tgt_embedder = self.embedders[_to]
        src_embedder_optim = self.optims['src']
        tgt_embedder_optim = self.optims['tgt']
        losses = []
        for batch in tqdm(self.train_dataloader):
            batch = utils.prepare_batch(batch, sw2i, tw2i)
            inputs, targets, input_lengths, target_lengths =\
                utils.pad_to_batch(batch, sw2i, tw2i)

            start_decode =\
                Variable(LT([[tw2i['<s>']] * targets.size(0)])).transpose(0, 1)
            self.encoder.zero_grad()
            self.decoder.zero_grad()
            src_embedder.zero_grad()
            tgt_embedder.zero_grad()

            if self.args.use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
                start_decode = start_decode.cuda()

            output, hidden_c = self.encoder(src_embedder,
                                            inputs,
                                            input_lengths)

            preds = self.decoder(tgt_embedder,
                                 start_decode,
                                 hidden_c,
                                 targets.size(1),
                                 output,
                                 None,
                                 True)
            loss = self.loss_func(preds, targets.view(-1))
            losses.append(loss.data[0])
            loss.backward()
            nn.utils.clip_grad_norm(self.encoder.parameters(), 50.0)
            nn.utils.clip_grad_norm(self.decoder.parameters(), 50.0)
            self.enc_optim.step()
            self.dec_optim.step()
            src_embedder_optim.step()
            tgt_embedder_optim.step()
        print(np.mean(losses))
        preds = preds.view(inputs.size(0), targets.size(1), -1)
        preds_max = torch.max(preds, 2)[1]
        print(' '.join([ti2w[p] for p in preds_max.data[0].tolist()]))
        print(' '.join([ti2w[p] for p in preds_max.data[1].tolist()]))

    def train_one_epoch_autoencoder(self, obj):
        print('objective: %s' % obj)
        sw2i = self.converters['src']['w2i']
        tw2i = self.converters['tgt']['w2i']
        i2w = self.converters[obj]['i2w']
        embedder = self.embedders[obj]
        embedder_optim = self.optims[obj]
        losses = []
        for batch in tqdm(self.train_dataloader):
            batch = utils.prepare_batch(batch, sw2i, tw2i)
            if obj == 'src':
                inputs, _, input_lengths, _ =\
                    utils.pad_to_batch(batch, sw2i, tw2i)
            if obj == 'tgt':
                _, inputs, _, input_lengths =\
                    utils.pad_to_batch(batch, sw2i, tw2i)

            start_decode =\
                Variable(LT([[tw2i['<s>']] * inputs.size(0)])).transpose(0, 1)
            self.encoder.zero_grad()
            self.decoder.zero_grad()
            embedder.zero_grad()

            if self.args.use_cuda:
                inputs = inputs.cuda()
                start_decode = start_decode.cuda()

            output, hidden_c = self.encoder(embedder,
                                            inputs,
                                            input_lengths)

            preds = self.decoder(embedder,
                                 start_decode,
                                 hidden_c,
                                 inputs.size(1),
                                 output,
                                 None,
                                 True)
            loss = self.loss_func(preds, inputs.view(-1))
            losses.append(loss.data[0])
            loss.backward()
            nn.utils.clip_grad_norm(self.encoder.parameters(), 50.0)
            nn.utils.clip_grad_norm(self.decoder.parameters(), 50.0)
            self.enc_optim.step()
            self.dec_optim.step()
            embedder_optim.step()
        print(np.mean(losses))
        preds = preds.view(inputs.size(0), inputs.size(1), -1)
        preds_max = torch.max(preds, 2)[1]
        print(' '.join([i2w[p] for p in preds_max.data[0].tolist()]))
        print(' '.join([i2w[p] for p in preds_max.data[1].tolist()]))
