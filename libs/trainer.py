import numpy as np
import torch
from torch.autograd import Variable
from torch import LongTensor as LT
import torch.optim as optim
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

        self.sw2i = self.train_dataloader.dataset.sw2i
        self.si2w = self.train_dataloader.dataset.si2w
        self.tw2i = self.train_dataloader.dataset.tw2i
        self.ti2w = self.train_dataloader.dataset.ti2w
        vocab_objs = {
                'tgt': {
                    # 'vocab': self.tgt_vocab,
                    'w2i': self.tw2i,
                    'i2w': self.ti2w
                    },
                'src': {
                    # 'vocab': self.src_vocab,
                    'w2i': self.sw2i,
                    'i2w': self.si2w
                    }
                }
        self.vocab_objs = vocab_objs

        # model
        encoder = models.Encoder(len(self.sw2i),
                                 args.src_embedding_size,
                                 args.encoder_hidden_n,
                                 args.encoder_num_layers,
                                 args.encoder_bidirectional,
                                 args.use_cuda)
        decoder = models.Decoder(len(self.tw2i),
                                 args.tgt_embedding_size,
                                 args.decoder_hidden_n,
                                 args.decoder_num_layers,
                                 args.decoder_dropout_p,
                                 args.use_cuda)
        src_embedder = models.Embedder(len(self.sw2i),
                                       args.src_embedding_size,
                                       self.sw2i['<PAD>'])
        tgt_embedder = models.Embedder(len(self.tw2i),
                                       args.tgt_embedding_size,
                                       self.tw2i['<PAD>'])
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

        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=0)

    def train_one_epoch(self, log_dict):
        sw2i = self.sw2i
        tw2i = self.tw2i
        ti2w = self.ti2w
        self.encoder.train()
        self.decoder.train()
        self.src_embedder.train()
        self.tgt_embedder.train()

        losses = []
        for batch in tqdm(self.train_dataloader):
            batch = utils.prepare_batch(batch, sw2i, tw2i)
            inputs, targets, input_lengths, target_lengths =\
                utils.pad_to_batch(batch, sw2i, tw2i)
            start_decode =\
                Variable(LT([[tw2i['<s>']] * targets.size(0)])).transpose(0, 1)

            self.encoder.zero_grad()
            self.decoder.zero_grad()
            self.src_embedder.zero_grad()
            self.tgt_embedder.zero_grad()

            if self.args.use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
                start_decode = start_decode.cuda()

            output, hidden_c = self.encoder(self.src_embedder,
                                            inputs,
                                            input_lengths)
            preds = self.decoder(self.tgt_embedder,
                                 start_decode,
                                 hidden_c,
                                 targets.size(1),
                                 output,
                                 None,
                                 True)
            loss = self.loss_func(preds,
                                  targets.view(-1))
            loss.backward()
            self.enc_optim.step()
            self.dec_optim.step()
            self.src_emb_optim.step()
            self.tgt_emb_optim.step()
            log_dict['train_losses'].append(loss.data[0])
            losses.append(loss.data[0])
        print('train loss %f' % np.mean(losses))
        preds = preds.view(inputs.size(0), targets.size(1), -1)
        preds_max = torch.max(preds, 2)[1]
        print(' '.join([ti2w[p] for p in preds_max.data[0].tolist()]))
        print(' '.join([ti2w[p] for p in preds_max.data[1].tolist()]))

    def translation_validate(self):
        args = self.args
        sw2i = self.sw2i
        tw2i = self.tw2i
        self.encoder.eval()
        self.decoder.eval()

        losses = []
        accs = []
        for batch in tqdm(self.test_dataloader):

            src_sents = batch['src']
            tgt_sents = batch['tgt']
            src_sents = [utils.convert_s2i(sent, sw2i)
                         for sent in src_sents]
            tgt_sents = [utils.convert_s2i(sent, tw2i)
                         for sent in tgt_sents]
            src_sents, tgt_sents, src_lens, tgt_lens =\
                utils.pad_batch(src_sents,
                                tgt_sents,
                                sw2i['<PAD>'],
                                tw2i['<PAD>'])

            src_sents = Variable(torch.LongTensor(src_sents), volatile=True)
            tgt_sents = Variable(torch.LongTensor(tgt_sents), volatile=True)
            if args.use_cuda:
                src_sents = src_sents.cuda()
                tgt_sents = tgt_sents.cuda()

            outputs, hidden = self.encoder(self.src_embedder,
                                           src_sents,
                                           src_lens)
            preds = self.decoder(self.tgt_embedder,
                                 self.tw2i['<s>'],
                                 outputs,
                                 hidden,
                                 tgt_sents.size(1))

            loss = self.loss_func(preds,
                                  tgt_sents.view(-1))
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
        ti2w = self.ti2w
        print('prediction: %s' % [ti2w[i.data[0]] for i in preds[0]])
        print('test loss %f' % np.mean(losses))
        print('test accuracy %f' % np.mean(accs))

    def autoencoder(self, subj):
        '''
        in:
          subj: string src or tgt
        '''
        args = self.args
        sw2i = self.sw2i
        tw2i = self.tw2i

        losses = []
        for batch in tqdm(self.train_dataloader):

            self.encoder.zero_grad()
            self.decoder.zero_grad()

            src_sents = batch['src']
            tgt_sents = batch['tgt']
            src_sents = [utils.convert_s2i(sent, sw2i)
                         for sent in src_sents]
            tgt_sents = [utils.convert_s2i(sent, tw2i)
                         for sent in tgt_sents]
            src_sents, tgt_sents, src_lens, tgt_lens =\
                utils.pad_batch(src_sents,
                                tgt_sents,
                                sw2i['<PAD>'],
                                tw2i['<PAD>'])

            src_sents = Variable(torch.LongTensor(src_sents))
            tgt_sents = Variable(torch.LongTensor(tgt_sents))
            if args.use_cuda:
                src_sents = src_sents.cuda()
                tgt_sents = tgt_sents.cuda()

            outputs, hidden = self.encoder(self.src_embedder,
                                           src_sents,
                                           src_lens)
            preds = self.decoder(self.tgt_embedder,
                                 self.tw2i['<s>'],
                                 outputs,
                                 hidden,
                                 tgt_sents.size(1))
            loss = self.loss_func(preds,
                                  tgt_sents.view(-1))
            loss.backward()
            self.enc_optim.step()
            self.dec_optim.step()
            losses.append(loss.data[0])
        print(np.mean(losses))
