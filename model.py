import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch


class HierarchicalSoftmax(nn.Module):
    def __init__(self, ntokens, nhid, ntokens_per_class = None):
        super(HierarchicalSoftmax, self).__init__()

        # Parameters
        self.ntokens = ntokens
        self.nhid = nhid

        if ntokens_per_class is None:
            ntokens_per_class = int(np.ceil(np.sqrt(ntokens)))

        self.ntokens_per_class = ntokens_per_class

        self.nclasses = int(np.ceil(self.ntokens * 1. / self.ntokens_per_class))
        self.ntokens_actual = self.nclasses * self.ntokens_per_class

        self.layer_top_W = nn.Parameter(torch.FloatTensor(self.nhid, self.nclasses), requires_grad=True)
        self.layer_top_b = nn.Parameter(torch.FloatTensor(self.nclasses), requires_grad=True)

        self.layer_bottom_W = nn.Parameter(torch.FloatTensor(self.nclasses, self.nhid, self.ntokens_per_class), requires_grad=True)
        self.layer_bottom_b = nn.Parameter(torch.FloatTensor(self.nclasses, self.ntokens_per_class), requires_grad=True)

        self.softmax = nn.Softmax(dim=1)

        self.init_weights()

    def init_weights(self):

        initrange = 0.1
        self.layer_top_W.data.uniform_(-initrange, initrange)
        self.layer_top_b.data.fill_(0)
        self.layer_bottom_W.data.uniform_(-initrange, initrange)
        self.layer_bottom_b.data.fill_(0)


    def forward(self, inputs, labels = None):

        batch_size, d = inputs.size()

        if labels is not None:

            label_position_top = labels / self.ntokens_per_class
            label_position_bottom = labels % self.ntokens_per_class

            layer_top_logits = torch.matmul(inputs, self.layer_top_W) + self.layer_top_b
            layer_top_probs = self.softmax(layer_top_logits)

            layer_bottom_logits = torch.squeeze(torch.bmm(torch.unsqueeze(inputs, dim=1), self.layer_bottom_W[label_position_top]), dim=1) + self.layer_bottom_b[label_position_top]
            layer_bottom_probs = self.softmax(layer_bottom_logits)

            target_probs = layer_top_probs[torch.arange(batch_size).long(), label_position_top] * layer_bottom_probs[torch.arange(batch_size).long(), label_position_bottom]

            return target_probs

        else:
            # Remain to be implemented
            layer_top_logits = torch.matmul(inputs, self.layer_top_W) + self.layer_top_b
            layer_top_probs = self.softmax(layer_top_logits)
            #print("CCC")
            #print(torch.sum(layer_top_probs))
            #print("QQQ")
            #print(layer_top_probs)
            #print(layer_top_probs[:,0])

            #print(layer_top_probs)
            #print(self.softmax(torch.matmul(inputs, self.layer_bottom_W[0]) + self.layer_bottom_b[0]))

            word_probs = layer_top_probs[:,0] * self.softmax(torch.matmul(inputs, self.layer_bottom_W[0]) + self.layer_bottom_b[0])
            #print(word_probs)
            for i in range(1, self.nclasses):
                word_probs = torch.cat((word_probs, layer_top_probs[:,i] * self.softmax(torch.matmul(inputs, self.layer_bottom_W[i]) + self.layer_bottom_b[i])), dim=1)

            #torch.bmm(inputs.unsqueeze(0).repeat(self.nclasses, 1, 1), self.layer_bottom_W) + self.layer_bottom_b


            # inputs.unsqueeze(1).repeat(1, self.nclasses, 1): batch_size x nclasses x nhid
            #inputs.unsqueeze(1).repeat(1, self.nclasses, 1).view(batch_size, -1)
            #print(layer_top_probs)
            #print("DDD")
            #print(torch.sum(word_probs))

            return word_probs





class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        #self.decoder = nn.Linear(nhid, ntoken)

        self.decoder = HierarchicalSoftmax(ntoken, nhid)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        #if tie_weights:
        #    if nhid != ninp:
        #        raise ValueError('When using the tied flag, nhid must be equal to emsize')
        #    self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        #self.decoder.bias.data.fill_(0)
        #self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, targets = None):
        if targets is not None:

            emb = self.drop(self.encoder(input))
            output, hidden = self.rnn(emb, hidden)
            output = self.drop(output)

            target_probs = self.decoder(output.view(-1, output.size(2)), targets)

            return target_probs, hidden

        else:
            # Remain to be implemented
            emb = self.drop(self.encoder(input))
            output, hidden = self.rnn(emb, hidden)
            output = self.drop(output)

            #print(input)
            #print(output)

            word_probs = self.decoder(output.view(-1, output.size(2)))

            return word_probs, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
