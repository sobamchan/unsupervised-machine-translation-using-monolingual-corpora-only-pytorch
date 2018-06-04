import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import LongTensor
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn.functional as F


class Embedder(nn.Module):

    def __init__(self, input_size, embedding_size, use_cuda=True):
        super(Embedder, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.use_cuda = use_cuda
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.embedding.weight = nn.init.xavier_uniform(self.embedding.weight)

    def forward(self, x):
        return self.embedding(x)


class Encoder(nn.Module):

    def __init__(self, input_size, embedding_size,
                 hidden_size, n_layers=1, bidirec=False, use_cuda=True):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.use_cuda = use_cuda

        if bidirec:
            self.n_direction = 2
            self.gru = nn.GRU(embedding_size,
                              hidden_size,
                              n_layers,
                              batch_first=True,
                              bidirectional=True)
        else:
            self.n_direction = 1
            self.gru = nn.GRU(embedding_size,
                              hidden_size,
                              n_layers,
                              batch_first=True)

    def init_hidden(self, inputs):
        hidden = Variable(torch.zeros(self.n_layers * self.n_direction,
                                      inputs.size(0),
                                      self.hidden_size))
        return hidden.cuda() if self.use_cuda else hidden

    def init_weight(self):
        self.gru.weight_hh_l0 = nn.init.xavier_uniform(self.gru.weight_hh_l0)
        self.gru.weight_ih_l0 = nn.init.xavier_uniform(self.gru.weight_ih_l0)

    def forward(self, embedder, inputs, input_lengths):
        hidden = self.init_hidden(inputs)
        embedded = embedder(inputs)
        packed = pack_padded_sequence(embedded,
                                      input_lengths,
                                      batch_first=True)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = pad_packed_sequence(outputs,
                                                      batch_first=True)

        if self.n_layers > 1:
            if self.n_direction == 2:
                hidden = hidden[-2:]
            else:
                hidden = hidden[-1]
        return outputs, torch.cat([h for h in hidden], 1).unsqueeze(1)


class Decoder(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size,
                 n_layers=1, dropout_p=0.1, use_cuda=True):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.use_cuda = use_cuda

        self.dropout = nn.Dropout(dropout_p)

        self.gru = nn.GRU(embedding_size + hidden_size,
                          hidden_size,
                          n_layers,
                          batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, input_size)
        self.attn = nn.Linear(self.hidden_size, self.hidden_size)

    def init_hidden(self, inputs):
        hidden = Variable(torch.zeros(self.n_layers,
                                      inputs.size(0),
                                      self.hidden_size))
        return hidden.cuda() if self.use_cuda else hidden

    def init_weight(self):
        self.gru.weight_hh_l0 = nn.init.xavier_uniform(self.gru.weight_hh_l0)
        self.gru.weight_ih_l0 = nn.init.xavier_uniform(self.gru.weight_ih_l0)
        self.linear.weight = nn.init.xavier_uniform(self.linear.weight)
        self.attn.weight = nn.init.xavier_uniform(self.attn.weight)

    def Attention(self, hidden, encoder_outputs, encoder_masking):
        hidden = hidden[0].unsqueeze(2)
        batch_size = encoder_outputs.size(0)
        max_len = encoder_outputs.size(1)
        energies = self.attn(encoder_outputs.contiguous()
                             .view(batch_size * max_len, -1))
        energies = energies.view(batch_size, max_len, -1)
        attn_energies = energies.bmm(hidden).squeeze(2)

        alpha = F.softmax(attn_energies, 1)
        alpha = alpha.unsqueeze(1)
        context = alpha.bmm(encoder_outputs)

        return context, alpha

    def forward(self, embedder, inputs, context, max_length, encoder_outputs,
                encoder_masking=False, is_training=False):
        embedded = embedder(inputs)
        hidden = self.init_hidden(inputs)
        if is_training:
            embedded = self.dropout(embedded)

        decode = []
        for i in range(max_length):
            _, hidden = self.gru(torch.cat((embedded, context), 2), hidden)
            if self.n_layers > 1:
                new_hidden = hidden[-1].unsqueeze(0)
            else:
                new_hidden = hidden
            concated = torch.cat((new_hidden, context.transpose(0, 1)), 2)
            score = self.linear(concated.squeeze(0))
            softmaxed = F.log_softmax(score, 1)
            decode.append(softmaxed)
            decoded = softmaxed.max(1)[1]
            embedded = embedder(decoded).unsqueeze(1)
            if is_training:
                embedded = self.dropout(embedded)
            context, alpha = self.Attention(new_hidden,
                                            encoder_outputs,
                                            encoder_masking)

        scores = torch.cat(decode, 1)
        return scores.view(inputs.size(0) * max_length, -1)

    def decode(self, embedder, context, encoder_outputs, w2i):
        start_decode = Variable(LongTensor([[w2i['<s>']] * 1])).transpose(0, 1)
        embedded = embedder(start_decode)
        hidden = self.init_hidden(start_decode)

        decodes = []
        attentions = []
        decoded = embedded
        while decoded.data.tolist()[0] != w2i['</s>']:
            _, hidden = self.gru(torch.cat((embedded, context), 2), hidden)
            concated = torch.cat((hidden, context.transpose(0, 1)), 2)
            score = self.linear(concated.squeeze(0))
            softmaxed = F.log_softmax(score, 1)
            decodes.append(softmaxed)
            decoded = softmaxed.max(1)[1]
            embedded = embedder(decoded).unsqueeze(1)
            context, alpha = self.Attention(hidden, encoder_outputs, None)
            attentions.append(alpha.squeeze(1))

        return torch.cat(decodes).max(1)[1], torch.cat(attentions)


class Discriminator(nn.Module):

    def __init__(self, in_d):
        super(Discriminator, self).__init__()
        self.in_d = in_d

        # layers = [nn.Dropout()]
        layers = []
        for i in range(3):
            input_dim = in_d if i == 0 else 1024
            output_dim = 1 if i == 2 else 1024
            layers.append(nn.Linear(input_dim, output_dim))
            if i < 2:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(0.1))
        layers.append(nn.Sigmoid())
        self.layers = layers

    def forward(self, x):
        return self.layers(x)
