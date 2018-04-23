import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import matplotlib .pyplot as plt
import pickle
from collections import Counter

OFFSET = '**OFFSET**'
START_TAG = '<sos>'
END_TAG = '<eos>'
UNK = '<UNK>'
def create_vocab(input_file):
    with open(input_file,'r',encoding='utf8') as f:
        inp=f.readlines()
        inp=[x.lower()  if x[-1]!='\n' else x[:-1] for x in inp]
        inp1=set("".join(inp).split(" "))
        inp1.remove("")
        char_set=set("".join(inp))
        return list(inp1),list(char_set)

def most_common_words(input_file):
    with open(input_file,'r') as f:
        inp=f.readlines()
        lines="".join(inp).split(" ")
        lines.remove("\n")
        lines=[x.replace("\n","") if x.startswith("\n") else x for x in lines]
        lines.remove("")
        c=Counter(lines)
        vocab=[x[0] for x in c.most_common()]
        return vocab
def create_word_to_ix(vocab):
     word_to_ix = {word:i+1 for i,word in enumerate(vocab)}
     word_to_ix[START_TAG]=0
     word_to_ix[END_TAG]=len(word_to_ix)+1
     word_to_ix[UNK]=len(word_to_ix)+1
     ix_to_word=  {i+1:word for i,word in enumerate(vocab)}
     ix_to_word[0] = START_TAG
     ix_to_word[len(ix_to_word)+1]=END_TAG
     ix_to_word[len(ix_to_word)+1]=UNK
     return word_to_ix,ix_to_word


def create_char_to_ix(vocab):
    char_to_ix={}
    chars="".join(vocab)
    chars=set(chars)
    chars=sorted(list(chars))
    for char in chars:
        char_to_ix[char]=len(char_to_ix)
    char_to_ix[UNK]=len(char_to_ix)
    return char_to_ix

def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()

def prepare_char_sequence(w,char_to_ix):
    #print(char_to_ix[w[0]])
    chr_idxs=[char_to_ix[char] if char in char_to_ix else char_to_ix[UNK] for char in w ]
    return chr_idxs

def prepare_target(seq,to_ix):
    try:
        return Variable(torch.LongTensor([to_ix[word] if word in to_ix else to_ix[UNK] for word in seq]))
    except Exception as e:
        pass
def prepare_sequence(seq, to_ix,char_to_ix):
    seq=seq[:-1]
    seq=seq.strip(" ")
    seq=seq.split(" ")

    try:
        idxs = [to_ix[word] if word in to_ix else to_ix[UNK] for word in seq]
    except Exception as e:
        pass
    #idxs=[(to_ix[word],prepare_char_sequence(word,char_to_ix)) if word in to_ix else (to_ix[UNK],prepare_char_sequence(word,char_to_ix)) for word in seq]
    idxs.append(to_ix[END_TAG])
    #print(idxs)
    tensor = torch.LongTensor(idxs)
    return Variable(tensor)

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def obtain_polyglot_embeddings(filename, word_to_ix):
    vecs = pickle.load(open(filename, 'rb'), encoding='latin1')

    vocab = [k for k, v in word_to_ix.items() if k!=START_TAG and k!=END_TAG ]

    word_vecs = {}
    for i, word in enumerate(vecs[0]):
        if word in word_to_ix:
            word_vecs[word] = np.array(vecs[1][i])

    word_embeddings = []
    word_embeddings.append(np.random.rand(64))
    for word in vocab:
        if word in word_vecs:
            embed = word_vecs[word]
        else:
            embed = vecs[1][0]
        word_embeddings.append(embed)
    word_embeddings.append(np.random.rand(64))
    word_embeddings.append(word_vecs[UNK])
    word_embeddings = np.array(word_embeddings)
    return word_embeddings