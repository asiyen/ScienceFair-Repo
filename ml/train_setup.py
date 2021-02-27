import torch
from torch import nn
import sys
from collections import defaultdict
import string
import pandas as pd


#seeds the rng engine so that results are reproducable
torch.manual_seed(1)

if len(sys.argv)==1:
    device=torch.device('cpu')
else:
    device=torch.device(sys.argv[-1])

#shorthand for sys.stdout.write
def p(thing):
    sys.stdout.write(thing)


def isnan(x):
    if isinstance(x, str):
        return False
    try:
        int(x)
        return False
    except:
        return True


def read_scraped(fname):
    res = []
    with open(fname, 'r') as f:
        for idx,i in enumerate(f):
            try:
                text = i[:-2]
                #print(text)
                label = torch.tensor(float(i.split(',')[-1])).reshape(-1).to(device)
                res.append((text, label))
            except Exception as e:
                #print("error occured in line ",idx,e)
                pass
    return res




#progress bar implementation
class fbar:
    def __init__(self, length):
        self.length = length - 1
        self.count = 0
        self.slen = 0

    def step(self):
        if self.count == self.length:
            p('\b' * self.slen)
            print(f'{self.count}/{self.length}')
            return
        p('\b' * self.slen)
        s = f'{self.count}/{self.length} '
        self.slen = len(s)
        p(s)
        self.count += 1


def within(x, y, margin):
    return abs(x - y) < margin

#tries to make data as even as possible by remixing data until desired randomness is achieved
def make_even_data(x, train_num, val_num,train_margin=0.05,val_margin=None,max_iters=7):
    if val_margin is None:
        val_margin=train_margin
    iters=0
    for i in range(max_iters):
        train, val = torch.utils.data.random_split(x, [train_num, val_num])
        print(
            get_fake(train),
            get_fake(val),
            'loop',
            abs(get_fake(train) - 0.5),
            abs(get_fake(val)-0.5))
        if within(get_fake(train), 0.25, train_margin) and within(
                get_fake(val), 0.25, val_margin):
            return train, val
    print('max iters reached')
    return train,val

# checks how many entries of data is marked as "fake" in a dataset
def get_fake(x):
    f = 0
    for i, j in x:
        if j == 0:
            f += 1
    return f / len(x)




# Experimental: batches text data
def batch_data(data, batch_size):
    result = []
    for i in range(len(data) // batch_size):
        start = batch_size * i
        stop = start + batch_size
        text = []
        labels = []
        for b in data[start:stop]:
            text.append(b[0])
            labels.append(b[1])
        result.append((text, torch.cat(labels)))
    text_last = []
    labels_last = []
    for b in data[start:-1]:
        text_last.append(b[0])
        labels_last.append(b[1])
    result.append((text_last, torch.cat(labels_last)))
    return result
from_folder=False
#main data reading section
try:
    newsdf = pd.read_csv('./data_and_models/corona_fake.csv')
    titles = newsdf['title']
    text = newsdf['text']
    labels = newsdf['label']
    tensorlabel = []
    for label in labels:
        if label == 'Fake' or label == 'fake':
            tensorlabel.append(torch.zeros(1).to(device))
        elif label == 'TRUE':
            tensorlabel.append(torch.ones(1).to(device))
        else:
            # print(label,'n')
            tensorlabel.append(-1)
    #filters out nan values from data
    all_text = [(text[i], tensorlabel[i]) for i in range(len(text)) if not isnan(text[i]) and tensorlabel[i] > -1]
    all_titles = [(titles[i], tensorlabel[i]) for i in range(len(titles)) if not isnan(titles[i]) and tensorlabel[i] > -1]
    merged = all_text + all_titles
    pro_train_text, pro_val_text = make_even_data(all_text, len(all_text) - 264, 264)

    pro_train_titles, pro_val_titles = make_even_data(all_titles, len(all_titles) - 264, 264)

    pro_train = pro_train_text + pro_train_titles
    pro_val = pro_val_text + pro_train_titles
    aux_data = read_scraped("./data_and_models/shuffled_cat.txt")
    aux_train, aux_val = make_even_data(aux_data,len(aux_data)-1000,1000,max_iters=2)
    train=list(aux_train+pro_train_text+pro_train_titles)
    val=list(aux_val+pro_val_text+pro_train_titles)
    print(get_fake(pro_train), get_fake(pro_val), 'pro')
    print("aux data:",get_fake(aux_train),get_fake(aux_val))
    print("total: ",get_fake(train),get_fake(val))
    from_folder=True
except FileNotFoundError as e:
    print("files not found",e)
    
    pass
def write2file(fname,data):
    with open(fname,'w') as f:
        for i,j in data:
            label=int(j)
            f.write(f'{i},{label}')
if not from_folder:
    train=read_scraped('train.txt')
    val=read_scraped('val.txt')
letters = string.ascii_letters + ' !0123456789?'
val_fakes=get_fake(val)
train_fakes=get_fake(train)
fake_weight=1/(train_fakes**0.5)
real_weight=1/((1-train_fakes)**0.5)
weights={0:fake_weight,1:real_weight}
def enum1(x):
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


def stack(data):
    result = []
    for i in range(len(data)):
        try:
            result.append(word2tensor(data[i]))
        except BaseException:
            result.append(torch.zeros(1))
    return result


def greatest_len(x):
    great = len(x[0])
    for i in x:
        if len(i) > great:
            great = i
        return great





def load_scraped_data(fname):
    result = []
    with open(fname, 'r') as f:
        for i in f:
            label, text = i.split('|')
            if not isnan(label) and not isnan(text):
                result.append((text, torch.tensor(float(label))))
    return result


with open("./data_and_models/stopwords.txt") as st:
    for i in st:
        stopwords = eval(i)


class filterer:
    def __init__(self, corpus):
        self.corpus = corpus

    def __call__(self, x):
        return x not in self.corpus


def remove_stops(x, c=stopwords, tolist=False):
    if tolist:
        return list(filter(filterer(c), x))
    else:
        return filter(filterer(c), x)


def absv(f, delim):
    with open(f, 'r') as df:
        ldf = list(df)
        result = [[] for i in range(len(ldf[0].split(delim)))]
        for h, i in enumerate(ldf[1:]):
            split = i.split(delim)
            text, l = split
            result[0].append(text)
            result[1].append(torch.tensor(float(l)).reshape(-1))
    return result


#good_train = list(zip(*absv('./good_training.txt', '|')))
#good_validate = list(zip(*absv('./good_validation.txt', '|')))


def batcher(x, y, batch):
    c = batch
    re = []
    for i in range(0, len(x), batch):
        re.append((x[i:c], y[i:c]))
    return re


def already_train_better(model, trains, vals, epochs,
                         optim='sgd', lr=0.01, loss='bce'):
    global losses, valis
    try:
        losses = []
        valis = []
        
        if loss == 'bce':
            lossf = nn.BCELoss(reduction='none')
            def fil_func(x): return x
        elif loss == 'logs':
            lossf = nn.BCEWithLogitsLoss()
            def fil_func(x): return torch.sigmoid(x)
        else:
            lossf = nn.MSELoss()
            def fil_func(x): return x
        if optim == 'sgd':
            o = torch.optim.SGD(model.parameters(), lr)
        elif optim == 'adam':
            o = torch.optim.Adam(model.parameters(), lr)
        elif optim == 'sadam':
            o = torch.optim.SparseAdam(model.parameters(), lr)
        for epoch in range(epochs):
            p = fbar(len(trains))
            avgloss = 0
            for i, j in trains:
                p.step()
                pred = model(i)
                target = j
                mask=torch.zeros_like(target)
                mask[target==0]=weights[0]
                mask[target==1]=weights[1]
                loss = (lossf(pred, target.reshape(-1))*weights[int(target)]).mean()
                
                model.zero_grad()
                o.zero_grad()
                loss.backward()
                o.step()
                avgloss += loss.data
            model.eval()
            with torch.no_grad():
                count = 0
                for vi, vt in vals:
                    pred = model(vi)
                    if torch.round(fil_func(pred)) == vt:
                        count += 1
            losses.append(avgloss / len(trains))
            valis.append(count / len(vals))
            model.train()
            print(count / len(vals), f' epoch: {epoch}')
            print(avgloss / len(trains))
        torch.save(model,f'model{str(losses[-1])[:5]}-{str(valis[-1])[:5]}')
        return model
    except KeyboardInterrupt:
        torch.save(model,f'model{str(losses[-1])[:5]}-{str(valis[-1])[:5]}')
        return model


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
        keys = keys / (k**0.25)
        queries = queries / (k**0.25)
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


wideatten = Thiccatten


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
        attended = self.transformers(pos_embeddings + w)


class native_transformer(nn.Module):
    def __init__(self, heads=8, depth=7, word_embed=20,
                 max_seq=6000, mode='mean'):
        super().__init__()
        self.transformers = torch.jit.script(
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=word_embed,
                    nhead=heads),
                depth))
        self.w_embed = nn.EmbeddingBag(len(vocab) + 1, word_embed, mode=mode)
        self.pos_embed = nn.Embedding(max_seq + 1, word_embed)
        self.fc = torch.jit.script(nn.Linear(word_embed, 1))
        self.sig = torch.jit.script(nn.Sigmoid())
        with open('./data_and_models/stopwords.txt', 'r') as df:
            for i in df:
                self.stopwords = eval(i)

    def forward(self, x):
        removed = [i for i in x.split(' ') if i not in self.stopwords]
        w = torch.stack([self.w_embed(word2tensor(i).unsqueeze(0))
                         for i in removed]).transpose(0, 1).to(device)
        b, t, k = w.size()
        pos_embeddings = self.pos_embed(torch.arange(t)).expand(b, t, k)
        attended = self.transformers(pos_embeddings + w)
        classes = self.fc(attended).mean(dim=1)
        return self.sig(classes.reshape(b, -1))


class batch_transformer(c_transformer):
    def forward(self, words):
        pieces = []
        for x in words:
            removed = [i for i in x.split(' ') if i not in self.stopwords]
            w = torch.stack([self.w_embed(word2tensor(i).to(device).unsqueeze(0))
                             for i in removed]).transpose(0, 1).to(device)
            pieces.append(w.squeeze(0))  # shape seq,embed
        padded = nn.utils.rnn.pad_sequence(pieces, batch_first=True)
        b, t, k = padded.size()
        pos_embeddings = self.pos_embed(torch.arange(t).to(device)).expand(b, t, k)
        attended = self.transformers(pos_embeddings + w)
        classes = self.fc(attended).mean(dim=1)
        return self.sig(classes.reshape(b, -1))
