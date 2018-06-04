import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import LongTensor as LT
from torch.autograd import Variable
from tqdm import tqdm
from libs import models
from libs import utils
from libs.dataset import get_dataloaders
from libs import sent_noise
from libs.word_translation_tools import bilingual_dictionary


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

        # Set bilingual dictionary
        self.bi_dict =\
            bilingual_dictionary.Dictionary(args.bilingual_dict_path)

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
        non_obj = 'src' if obj == 'tgt' else 'tgt'
        w2i = self.converters[obj]['w2i']
        i2w = self.converters[obj]['i2w']
        embedder = self.embedders[obj]
        embedder_optim = self.optims[obj]
        losses = []
        for batch in tqdm(self.train_dataloader):

            # add noise
            org_batch = copy.deepcopy(batch)
            batch[obj] = [sent_noise.run(s) for s in org_batch[obj]]
            batch[non_obj] = org_batch[obj]

            # convert string to ids
            batch = utils.prepare_batch(batch, w2i, w2i)

            inputs, targets, input_lengths, target_lengths =\
                utils.pad_to_batch(batch, w2i, w2i)

            start_decode =\
                Variable(LT([[w2i['<s>']] * inputs.size(0)])).transpose(0, 1)
            self.encoder.zero_grad()
            self.decoder.zero_grad()
            embedder.zero_grad()

            if self.args.use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
                start_decode = start_decode.cuda()

            output, hidden_c = self.encoder(embedder,
                                            inputs,
                                            input_lengths)

            preds = self.decoder(embedder,
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
            embedder_optim.step()
        print(np.mean(losses))
        preds = preds.view(inputs.size(0), targets.size(1), -1)
        preds_max = torch.max(preds, 2)[1]
        print(' '.join([i2w[p] for p in preds_max.data[0].tolist()]))
        print(' '.join([i2w[p] for p in preds_max.data[1].tolist()]))

    def train_one_epoch_cross_domain(self, obj, first_iter=False):
        non_obj = 'src' if obj == 'tgt' else 'tgt'
        print('Calculating cross domain loss %s to %s...' % (obj, non_obj))
        obj_w2i = self.converters[obj]['w2i']
        obj_i2w = self.converters[obj]['i2w']
        non_obj_w2i = self.converters[non_obj]['w2i']
        non_obj_i2w = self.converters[non_obj]['i2w']
        obj_embedder = self.embedders[obj]
        obj_embedder_optim = self.optims[obj]
        non_obj_embedder = self.embedders[non_obj]
        non_obj_embedder_optim = self.optims[non_obj]
        losses = []

        for batch in tqdm(self.train_dataloader):

            # translate obj to non_obj with previous iter model
            if first_iter:
                src_to_tgt = True if obj == 'src' else False
                naive_y =\
                    [' '.join(self.bi_dict.translate(sent.split(),
                                                     src_to_tgt=src_to_tgt))
                     for sent in batch[obj]]
            else:
                naive_y = self.translate(batch[obj])

            noised_y = [sent_noise.run(sent) for sent in naive_y]

            batch['tgt'] = batch[obj]
            batch['src'] = noised_y

            # convert string to ids
            batch = utils.prepare_batch(batch, non_obj_w2i, obj_w2i)

            inputs, targets, input_lengths, target_lengths =\
                utils.pad_to_batch(batch, non_obj_w2i, obj_w2i)

            start_decode =\
                Variable(LT([[non_obj_w2i['<s>']] * inputs.size(0)]))\
                .transpose(0, 1)
            self.encoder.zero_grad()
            self.decoder.zero_grad()
            obj_embedder.zero_grad()
            non_obj_embedder.zero_grad()

            if self.args.use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
                start_decode = start_decode.cuda()

            output, hidden_c = self.encoder(non_obj_embedder,
                                            inputs,
                                            input_lengths)

            preds = self.decoder(obj_embedder,
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
            obj_embedder_optim.step()
            non_obj_embedder_optim.step()

        print(np.mean(losses))
        preds = preds.view(inputs.size(0), targets.size(1), -1)
        preds_max = torch.max(preds, 2)[1]
        print(' '.join([non_obj_i2w[p] for p in inputs.data[0].tolist()]))
        print(' '.join([obj_i2w[p] for p in preds_max.data[0].tolist()]))
        print(' '.join([obj_i2w[p] for p in targets.data[0].tolist()]))

        print(' '.join([non_obj_i2w[p] for p in inputs.data[1].tolist()]))
        print(' '.join([obj_i2w[p] for p in preds_max.data[1].tolist()]))
        print(' '.join([obj_i2w[p] for p in targets.data[1].tolist()]))

    def clip_current_model(self):
        """
        Clip translation model (src -> tgt, tgt -> src) parameters
        to use in cross domain loss calculation
        """
        self.prev_encoder = copy.deepcopy(self.encoder)
        self.prev_decoder = copy.deepcopy(self.decoder)
        self.prev_embedders = {
                'src': copy.deepcopy(self.src_embedder),
                'tgt': copy.deepcopy(self.tgt_embedder),
                }
        return

    def translate(self, sents, obj):
        non_obj = 'src' if obj == 'tgt' else 'tgt'
        print('Translating %s -> %s...' % (non_obj, obj))
        sw2i = self.converters[non_obj]['w2i']
        tw2i = self.converters[obj]['w2i']
        ti2w = self.converters[obj]['i2w']

        src_embedder = self.prev_embedders[non_obj]
        tgt_embedder = self.prev_embedders[obj]
        encoder = self.prev_encoder
        decoder = self.prev_decoder

        batch = {'src': sents, 'tgt': sents}
        batch = utils.prepare_batch(batch, sw2i, tw2i)
        inputs, targets, input_lengths, target_lengths =\
            utils.pad_to_batch(batch, sw2i, tw2i)
        start_decode =\
            Variable(LT([[tw2i['<s>']] * targets.size(0)])).transpose(0, 1)
        if self.args.use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
            start_decode = start_decode.cuda()
        output, hidden_c = encoder(src_embedder,
                                   inputs,
                                   input_lengths)
        max_length = 50
        preds = decoder(tgt_embedder,
                        start_decode,
                        hidden_c,
                        max_length,
                        output,
                        None,
                        True)
        preds = preds.view(inputs.size(0), max_length, -1)
        preds_max = torch.max(preds, 2)[1]

        result_sents = []
        for i in range(len(sents)):
            result_sent =\
                ' '.join([ti2w[p] for p in preds_max.data[i].tolist()])
            result_sents.append(result_sent)
        return result_sents
