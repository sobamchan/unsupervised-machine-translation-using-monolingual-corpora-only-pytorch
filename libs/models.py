import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, vocab_n, embedding_dim, padding_idx, dropout_p,
                 hidden_n, num_layers, bidirectional, use_cuda):
        super(Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_n = hidden_n
        self.use_cuda = use_cuda
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_n, embedding_dim, padding_idx)
        self.dropout = nn.Dropout2d(dropout_p)
        self.gru = nn.GRU(embedding_dim,
                          hidden_n,
                          num_layers,
                          batch_first=True,
                          bidirectional=bidirectional)

    def init_hidden(self, inputs):
        dim = self.num_layers * 2 if self.bidirectional else self.num_layers
        batch_size = inputs.size(0)
        hidden = torch.zeros(dim, batch_size, self.hidden_n)
        hidden = Variable(hidden)
        return hidden.cuda() if self.use_cuda else hidden

    def forward(self, src_sents, src_lens):
        x = self.embedding(src_sents)
        x = self.dropout(x)
        hidden = self.init_hidden(x)
        x = pack_padded_sequence(x, src_lens, batch_first=True)
        outputs, hidden = self.gru(x, hidden)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)

        if self.num_layers > 1:
            if self.bidirectional:
                hidden = hidden[-2:]
            else:
                hidden = hidden[-1]

        return outputs, hidden


class Decoder(nn.Module):

    def __init__(self, vocab_n, embedding_dim, padding_idx, dropout_p,
                 hidden_n, num_layers, bidirectional, use_cuda):
        super(Decoder, self).__init__()
        self.use_cuda = use_cuda
        self.num_layers = num_layers
        self.hidden_n = hidden_n
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_n, embedding_dim, padding_idx)
        self.dropout = nn.Dropout2d(dropout_p)
        if self.bidirectional:
            gru_input_dim = embedding_dim + hidden_n * 2
        else:
            gru_input_dim = embedding_dim + hidden_n
        self.gru = nn.GRU(gru_input_dim,
                          hidden_n,
                          num_layers,
                          batch_first=True,
                          bidirectional=bidirectional)
        if bidirectional:
            self.proj = nn.Linear(hidden_n * 2 + hidden_n * 2, vocab_n)
            self.attention = nn.Linear(hidden_n * 2, hidden_n * 2)
        else:
            self.proj = nn.Linear(hidden_n * 2, vocab_n)
            self.attention = nn.Linear(hidden_n, hidden_n)

    def init_hidden(self, inputs):
        dim = self.num_layers * 2 if self.bidirectional else self.num_layers
        batch_size = inputs.size(0)
        hidden = torch.zeros(dim, batch_size, self.hidden_n)
        hidden = Variable(hidden)
        return hidden.cuda() if self.use_cuda else hidden

    def forward(self, start_id, enc_outputs, context, dec_max_len):
        '''
        in:
          start_id: int
          enc_outputs: B, S, H
          context: direction_n, B, H
        '''
        batch_size = enc_outputs.size(0)
        start = Variable(torch.LongTensor([[start_id]] * batch_size))
        if self.use_cuda:
            start = start.cuda()
        hidden = self.init_hidden(enc_outputs)
        embedded = self.embedding(start)
        context = context.transpose(0, 1)  # B, direction_n, H
        preds = []
        for i in range(dec_max_len):
            embedded = self.dropout(embedded)  # B, 1, H
            gru_input =\
                torch.cat([embedded, context], 1)  # B, 1 + direction_n, H
            gru_input = gru_input.view(batch_size,
                                       1,
                                       -1)  # B, 1, 1 + direction_n*H

            _, hidden = self.gru(gru_input,
                                 hidden)
            # hidden: num_layers*direction_n, B, H

            last_hidden = hidden[-2:]\
                if self.bidirectional else hidden[-1]
            last_hidden = last_hidden.transpose(0, 1)
            # hidden: B, direction_n, H

            context = context.contiguous()
            last_hidden = last_hidden.contiguous().view(batch_size, -1)
            cat = torch.cat((last_hidden, context.view(batch_size, -1)), 1)

            pred = self.proj(cat.contiguous().view(batch_size, -1))  # B, V
            pred = F.log_softmax(pred, 1)
            preds.append(pred)

            embedded = self.embedding(pred.max(1)[1]).unsqueeze(1)

            context, alpha = self.calc_attention(enc_outputs, last_hidden)
        preds = torch.cat(preds, 1).view(batch_size * dec_max_len, -1)
        return preds

    def calc_attention(self, enc_outputs, hidden):
        '''
        in:
          enc_outputs: B, S, H
          hidden: B, H*direction_n
        '''
        batch_size = enc_outputs.size(0)
        enc_outputs = enc_outputs.contiguous()
        hidden = hidden.unsqueeze(1)  # 1, B, H*direction_n
        batch_size, seq_len, _ = enc_outputs.size()
        eng = self.attention(
                enc_outputs.view(batch_size * seq_len, -1))  # B*S, H
        eng = eng.view(batch_size, seq_len, -1)  # B, S, H
        eng = torch.bmm(eng, hidden.transpose(1, 2))  # B, S, 1
        eng = eng.squeeze(2)  # B, S
        alpha = F.softmax(eng, 1)
        alpha = alpha.unsqueeze(1)  # B, 1, S
        context = torch.bmm(alpha, enc_outputs)  # B, 1, H
        if self.bidirectional:
            context = context.view(batch_size, 2, -1)
        return context, alpha
