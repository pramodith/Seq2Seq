import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import matplotlib .pyplot as plt
import pickle
import preprocessing
import random
from collections import OrderedDict

START_TAG = '<sos>'
END_TAG = '<eos>'
use_cuda=True
MAX_LENGTH=40

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size,embeddings):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.GRU(hidden_size, hidden_size)
        self.embedding.weight.data.copy_(torch.from_numpy(embeddings))
    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def initHidden(self):
        use_cuda=True
        result = (Variable(torch.randn(1, 1, self.hidden_size)))
        return result

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size/2,bidirectional=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        use_cuda=True
        result = (Variable(torch.randn(2, 1, self.hidden_size // 2)),
                Variable(torch.randn(2, 1, self.hidden_size // 2)))
        return result

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, embeddings,dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.embedding.weight.data.copy_(torch.from_numpy(embeddings))

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1,1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        use_cuda=True
        result =(Variable(torch.randn(1, 1, self.hidden_size)))
        return result
            

teacher_forcing_ratio = 0.5


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs if use_cuda else encoder_outputs

    loss = 0


    for ei in range(min(input_length,MAX_LENGTH)):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden)

        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([word_to_ix[START_TAG]]))
    decoder_input = decoder_input if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(min(target_length,max_length)):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(min(target_length,max_length)):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == END_TAG:
                break

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()
    return loss / target_length

import time

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    batch_size=32
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate,momentum=0.8)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate,momentum=0.8)
    with open("sample_input_soul.csv") as f:
        X=f.readlines()
    with open("sample_output_soul.csv") as f:
        Y=f.readlines()
    training_pairs = [[preprocessing.prepare_sequence(X[i],word_to_ix,None),preprocessing.prepare_sequence(Y[i],word_to_ix,None)]for i in range(n_iters)]
    #n_iters = int(n_iters / 32)
    criterion = nn.NLLLoss()
    #training_pairs=[training_pairs[i:i+32] for i in range(n_iters)]
    for epoch in range(50):
        print_loss_total=0
        print(epoch)
        for iter in range(1, n_iters + 1):
            #training_pair=training_pairs[iter*32:(iter+1)*32]
            training_pair = training_pairs[iter - 1]
            input_variable = training_pair[0]
            target_variable = training_pair[1]
            #input_variable = torch.LongTensor([ x[0] for x in training_pair])
            #target_variable = torch.LongTensor([x[1] for x in training_pair])

            loss = train(input_variable, target_variable, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss
        print("Total loss is %s",print_loss_total)
    return


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    input_variable = preprocessing.prepare_sequence(sentence,word_to_ix,None)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([word_to_ix[START_TAG]]))  # SOS
    decoder_input = decoder_input if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == word_to_ix[END_TAG]:
            decoded_words.append('<eos>')
            break
        else:
            decoded_words.append(ix_to_word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input if use_cuda else decoder_input

    #return decoded_words, decoder_attentions[:di + 1]
    return decoded_words
#vocab,characters=preprocessing.create_vocab("sample_input1.csv")
vocab=preprocessing.most_common_words("sample_input_soul.csv")
word_to_ix,ix_to_word=preprocessing.create_word_to_ix(vocab)
embeddings=preprocessing.obtain_polyglot_embeddings('polyglot-en.pkl',word_to_ix)
encoder = EncoderRNN(len(vocab)+4, 64,embeddings)
decoder = AttnDecoderRNN(64, len(vocab)+4,embeddings)
encoder.load_state_dict(torch.load('encoder_soul.pkl'))
decoder.load_state_dict(torch.load('decoder_soul.pkl'))
#for i in range(50):
#trainIters(encoder, decoder, 126)
#torch.save(encoder.state_dict(), "encoder_soul.pkl")
#torch.save(decoder.state_dict(), "decoder_soul.pkl")

while True:
    print("Enter a sentence to be evaluated \n")
    sentence=input()
    print(evaluate(encoder,decoder,sentence))
