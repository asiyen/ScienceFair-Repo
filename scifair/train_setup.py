import torch
from torch import nn
import string
from collections import defaultdict
import copy
torch.manual_seed(1)
with open('stopwords.txt') as stops:
    for i in stops:
        stopwords = eval(i)
letters = string.ascii_letters + ' !0123456789?'
device = torch.device('cpu')


def enum1(x
         ):
    for i in range(len(x)):
        yield (i + 1, x[i])


vocab = {letter: i for i, letter in enum1(letters)}
vocab = defaultdict(lambda: 0, vocab)


# print(vocab[''],vocab['7'])


def word2tensor(word):
    tens = [vocab[i] for i in word]
    if len(tens) > 0:
        return torch.tensor(tens)
    else:
        return torch.tensor([0])


def remove_stops(x):
    return filter(lambda i: i not in stopwords, x)


class Thiccatten(nn.Module):
    def __init__(self, k, heads):
        super().__init__()
        self.qw = nn.Linear(k, k * heads)
        self.kw = nn.Linear(k, k * heads)
        self.vw = nn.Linear(k, k * heads)
        self.fc = nn.Linear(k * heads, k)
        self.heads = heads

    def forward(self, x):
        b, t, k = x.size()
        h = self.heads
        q = self.qw(x).view(b, t, h, k)
        key = self.kw(x).view(b, t, h, k)
        v = self.vw(x).view(b, t, h, k)
        keys = key.transpose(1, 2).contiguous().view(b * h, t, k)
        queries = q.transpose(1, 2).contiguous().view(b * h, t, k)
        values = v.transpose(1, 2).contiguous().view(b * h, t, k)
        keys = keys / (k ** 0.25)
        queries = queries / (k ** 0.25)
        dot = torch.bmm(keys, queries.transpose(1, 2))
        scaled_dot = torch.softmax(dot, dim=2)
        out = torch.bmm(scaled_dot, values).view(b, h, t, k)
        out = out.transpose(1, 2).contiguous().view(b, t, k * h)
        return self.fc(out)


class tblock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()

        self.attention = Thiccatten(k, heads=heads)

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.ff = nn.Sequential(
            nn.Linear(k, 4 * k),
            nn.ReLU(),
            nn.Linear(4 * k, k))

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)

        fedforward = self.ff(x)
        return self.norm2(fedforward + x)



class c_transformer(nn.Module):
    def __init__(self, heads=8, depth=7, word_embed=20,
                 max_seq=6000, mode='mean'):
        super().__init__()
        self.transformers = nn.Sequential(
            *[tblock(word_embed, heads) for i in range(depth)])
        self.w_embed = nn.EmbeddingBag(len(vocab) + 1, word_embed, mode=mode)
        self.pos_embed = nn.Embedding(max_seq + 1, word_embed)
        self.fc = nn.Linear(word_embed, 1)

    def forward(self, x):
        w = torch.stack([self.w_embed(word2tensor(i).unsqueeze(0).to(device))
                         for i in remove_stops(x.split(' '))]).transpose(0, 1).to(device)
        b, t, k = w.size()
        pos_embeddings = self.pos_embed(torch.arange(t).to(device)).expand(b, t, k)
        attended = self.transformers(pos_embeddings + w)
        classes = self.fc(attended).mean(dim=1)
        return torch.sigmoid(classes.reshape(-1))
class duplicate_transformer(nn.Module):
    def __init__(self, heads=8, depth=7, word_embed=20,
                 max_seq=6000, mode='mean'):
        super().__init__()
        self.transformer_instance=tblock(word_embed,heads)
        self.transformers = nn.Sequential(
            *[copy.deepcopy(self.transformer_instance) for i in range(depth)])
        self.w_embed = nn.EmbeddingBag(len(vocab) + 1, word_embed, mode=mode)
        self.pos_embed = nn.Embedding(max_seq + 1, word_embed)
        self.fc = nn.Linear(word_embed, 1)

    def forward(self, x):
        w = torch.stack([self.w_embed(word2tensor(i).unsqueeze(0).to(device))
                         for i in remove_stops(x.split(' '))]).transpose(0, 1).to(device)
        b, t, k = w.size()
        pos_embeddings = self.pos_embed(torch.arange(t).to(device)).expand(b, t, k)
        attended = self.transformers(pos_embeddings + w)
        classes = self.fc(attended).mean(dim=1)
        return torch.sigmoid(classes.reshape(-1))

