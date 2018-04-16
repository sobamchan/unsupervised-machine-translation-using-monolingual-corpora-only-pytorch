import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class Encoder(nn.Module):

    def __init__(self, vocab, args):
        super(Encoder, self).__init__()
        self.args = args
        self.in_size = len(vocab)
        self.hidden_n = args.encoder_hidden_n
        self.layers_n = args.encoder_layers_n
        self.embedding_dim = args.encoder_embedding_dim

        self.embedding = nn.Embedding(self.in_size, self.embedding_dim)

        if args.encoder_bidirec:
            self.direction_n = 2
            self.gru = nn.GRU(self.embedding_dim,
                              self.hidden_n,
                              self.layers_n,
                              batch_first=True,
                              bidirectional=True)
        else:
            self.direction_n = 1
            self.gru = nn.GRU(self.embedding_dim,
                              self.hidden_n,
                              self.layers_n,
                              batch_first=True)

    def init_hidden(self, inputs):
        args = self.args
        hidden = Variable(torch.zeros(self.layers_n * self.direction_n,
                                      inputs.size(0),
                                      self.hidden_n))
        return hidden.cuda() if args.use_cuda else hidden

    def init_weight(self):
        self.embedding.weight = nn.init.xavier_uniform(self.embedding.weight)
        self.gru.weight_hh_l0 = nn.init.xavier_uniform(self.gru.weight_hh_l0)
        self.gru.weight_ih_l0 = nn.init.xavier_uniform(self.gru.weight_ih_l0)

    def forward(self, inputs, input_lens):
        hidden = self.init_hidden(inputs)
        embedded = self.embedding(inputs)
        packed = pack_padded_sequence(embedded,
                                      input_lens,
                                      batch_first=True)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lens = pad_packed_sequence(outputs, batch_first=True)

        if self.layers_n > 1:
            if self.direction_n == 2:
                hidden = hidden[-2:]
            else:
                hidden = hidden[-1]
        return outputs, torch.cat(list(hidden), 1).unsqueeze(1)


class Decoder(nn.Module):

    def __init__(self, vocab, args):
        super(Decoder, self).__init__()
        self.args = args
        self.hidden_n = args.decoder_hidden_n * 2
        self.layers_n = args.decoder_layers_n
        self.embedding_dim = args.decoder_embedding_dim
        self.vocab = vocab

        self.embedding = nn.Embedding(len(vocab), self.embedding_dim)
        self.dropout = nn.Dropout(args.dropout_p)
        self.gru = nn.GRU(self.embedding_dim + self.hidden_n,
                          self.hidden_n,
                          self.layers_n,
                          batch_first=True)
        self.linear = nn.Linear(self.hidden_n * 2, len(vocab))
        self.attention = nn.Linear(self.hidden_n, self.hidden_n)

    def init_hidden(self, inputs):
        hidden = Variable(torch.zeros(self.layers_n,
                                      inputs.size(0),
                                      self.hidden_n))
        return hidden.cuda() if self.args.use_cuda else hidden

    def init_weight(self):
        self.embedding.weight = nn.init.xavier_uniform(self.embedding.weight)
        self.gru.weight_hh_l0 = nn.init.xavier_uniform(self.gru.weight_hh_l0)
        self.gru.weight_ih_l0 = nn.init.xavier_uniform(self.gru.weight_ih_l0)
        self.linear.weight = nn.init.xavier_uniform(self.linear.weight)
        self.attention.weight = nn.init.xavier_uniform(self.attention.weight)

    def calc_attention(self, hidden, encoder_outputs):
        hidden = hidden[0].unsqueeze(2)
        batch_size = encoder_outputs.size(0)
        max_len = encoder_outputs.size(1)
        energies = self.attention(encoder_outputs.contiguous()
                                  .view(batch_size * max_len,
                                        -1))
        energies = energies.view(batch_size, max_len, -1)
        attention_energies = energies.bmm(hidden).squeeze(2)

        alpha = F.softmax(attention_energies, dim=1)
        alpha = alpha.unsqueeze(1)
        context = alpha.bmm(encoder_outputs)
        return context, alpha

    def forward(self, inputs, context, max_len, encoder_outputs):
        embedded = self.embedding(inputs)
        hidden = self.init_hidden(inputs)
        if self.training:
            embedded = self.dropout(embedded)

        decode = []
        for i in range(max_len):
            _, hidden = self.gru(torch.cat((embedded, context), 2),
                                 hidden)
            concated = torch.cat((hidden[-1].unsqueeze(0),
                                 context.transpose(0, 1)),
                                 2)
            score = self.linear(concated.squeeze(0))
            softmaxed = F.log_softmax(score, dim=1)
            decoded = softmaxed.max(1)[1]
            embedded = self.embedding(decoded).unsqueeze(1)
            if self.training:
                embedded = self.dropout(embedded)
            context, alpha = self.calc_attention(hidden, encoder_outputs)
            decode.append(softmaxed)
            del softmaxed

        scores = torch.cat(decode, 1)
        return scores.view(inputs.size(0) * max_len, -1)

    def decode(self, inputs, context, encoder_outputs):
        embedded = self.embedding(inputs)
        hidden = self.init_hidden(inputs)

        decodes = []
        attentions = []
        decoded = embedded
        for _ in range(50):
            _, hidden = self.gru(torch.cat((embedded,
                                            context), 2),
                                 hidden)  # h_t = f(h_{t-1}, y_{t-1}, c)
            concated = torch.cat((hidden[-1].unsqueeze(0),
                                 context.transpose(0, 1)),
                                 2)
            score = self.linear(concated.squeeze(0))
            softmaxed = F.log_softmax(score)
            decodes.append(softmaxed)
            decoded = softmaxed.max(1)[1]
            embedded = self.embedding(decoded).unsqueeze(1)
            context, alpha = self.calc_attention(hidden, encoder_outputs)
            attentions.append(alpha.squeeze(1))

        decodes = torch.cat(decodes, 1).view(inputs.size(0), 50, -1)
        return decodes.max(2)[1], torch.cat(attentions)
