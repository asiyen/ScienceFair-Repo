import torch
from torch import nn
import sys 
from collections import defaultdict
import string
import pandas as pd
def p(thing):
    sys.stdout.write(thing)


def isnan(x):
    if type(x) == str:
        return False
    try:
        int(x)
        return False
    except:
        return True

device=torch.device('cpu')
class pbar:
    def __init__(self, length, total, frac=False):
        self.length = length
        self.total = total
        if not frac:
            p('|' + '.' * length + '|')
        else:
            pass
        self.count = 0
        self.thresh = total // length
        self.all_count = 0
        self.n = 0

    def frac(self):
        if self.n == self.total - 1:
            p('\n')
            return
        if self.n > 0:
            p('\b' * self.len)
        string = f'{self.n}/{self.total}'
        p(string)
        self.len = len(string)
        self.n += 1


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

def within(x,y,margin):
  return abs(x-y)<margin
def make_even_data(x,train_num,val_num):
  while 1:
    train,val=torch.utils.data.random_split(x,[train_num,val_num])
    print(get_fake(train),get_fake(val),'loop',get_fake(train)-get_fake(val))
    if within(get_fake(train),0.5,0.03) and within(get_fake(val),0.5,0.03):
      return train,val
def get_fake(x):
  f=0
  for i,j in x:
    if j==0:
      f+=1
  return f/len(x)
try:
  newsdf = pd.read_csv('./corona_fake.csv')
  titles = newsdf['title']
  text = newsdf['text']
  labels = newsdf['label']
  tensorlabel = []
  for label in labels:
      if label == 'Fake' or label == 'fake':
          tensorlabel.append(torch.zeros(1))
      elif label == 'TRUE':
          tensorlabel.append(torch.ones(1))
      else:
          # print(label,'n')
          tensorlabel.append(-1)
  all_text=[(text[i],tensorlabel[i]) for i in range(len(text)) if not isnan(text[i]) and tensorlabel[i]>-1]
  all_titles=[(titles[i],tensorlabel[i]) for i in range(len(titles)) if not isnan(titles[i]) and tensorlabel[i]>-1]
  merged=all_text+all_titles
  pro_train,pro_val=make_even_data(merged,len(merged)-400,400)
  print(get_fake(pro_train),get_fake(pro_val),'pro')
except FileNotFoundError:
  pass

letters = string.ascii_letters + ' !0123456789?'

def enum1(x):
    for i in range(len(x)):
        yield (i + 1, x[i])


vocab = {letter: i for i, letter in enum1(letters)}
vocab = defaultdict(lambda: 0, vocab)
#print(vocab[''],vocab['7'])
def word2tensor(word):
    tens = [vocab[i] for i in word]
    if len(tens)>0:
      return torch.tensor(tens)
    else:
      return torch.tensor([0])


def stack(data):
    result = []
    for i in range(len(data)):
        try:
            result.append(word2tensor(data[i]))
        except:
            result.append(torch.zeros(1))
    return result





def greatest_len(x):
    great = len(x[0])
    for i in x:
        if len(i) > great:
            great = i
        return great


def pad_to(x, length):
  if len(x)>=length:
    return x
  return torch.nn.functional.pad(x, (0, length - len(x)))


def clean(thing, length):
    new = []
    for sentence in thing:
        new.append(pad_to(sentence, length))
    return new

def load_scraped_data(fname):
  result=[]
  with open(fname,'r') as f:
    for i in f:
      label,text=i.split('|')
      if not isnan(label) and not isnan(text):
        result.append((text,torch.tensor(float(label))))
  return result
#all_text=[(text[i],tensorlabel[i]) for i in range(len(text)) if not isnan(text[i]) and tensorlabel[i]>-1]
#all_titles=[(titles[i],tensorlabel[i]) for i in range(len(titles)) if not isnan(titles[i]) and tensorlabel[i]>-1]
#pro_train_text,pro_val_text=torch.utils.data.random_split(all_text,[len(all_text)-264,264])
with open('./stopwords.txt') as st:
  for i in st:
    stopwords=eval(i)
class filterer:
  def __init__(self,corpus):
    self.corpus=corpus
  def __call__(self,x):
    return x not in self.corpus
def remove_stops(x,c=stopwords,tolist=False):
  if tolist:
    return list(filter(filterer(c),x))
  else:
    return filter(filterer(c),x)
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
good_train=list(zip(*absv('./good_training.txt','|')))
good_validate=list(zip(*absv('./good_validation.txt','|')))
def batcher(x, y, batch):
    c = batch
    re = []
    for i in range(0, len(x), batch):
        re.append((x[i:c], y[i:c]))
    return re
def already_train_better(model,trains,vals,epochs,optim='sgd',lr=0.01,loss='bce'):
  global losses, valis
  try:
    losses=[]
    valis=[]
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if loss=='bce':
      lossf=nn.BCELoss()
      fil_func=lambda x: x
    elif loss=='logs':
      lossf=nn.BCEWithLogitsLoss()
      fil_func=lambda x:torch.sigmoid(x)
    else:
      lossf=nn.MSELoss()
      fil_func=lambda x:x
    if optim=='sgd':
      o=torch.optim.SGD(model.parameters(),lr)
    elif optim=='adam':
      o=torch.optim.Adam(model.parameters(),lr)
    elif optim=='sadam':
      o=torch.optim.SparseAdam(model.parameters(),lr)
    for epoch in range(epochs):
      p=fbar(len(trains))
      avgloss=0
      for i,j in trains:
        p.step()
        pred=model(i)
        target=j.float()
        loss=lossf(pred.float(),target)
        model.zero_grad()
        o.zero_grad()
        loss.backward()
        o.step()
        avgloss+=loss
      model.eval()
      with torch.no_grad():
        count=0
        for vi,vt in vals:
          pred=model(vi)
          if torch.round(fil_func(pred))==vt:
              count+=1
      losses.append(avgloss/len(trains))
      valis.append(count/len(vals))
      model.train()
      print(count/len(vals),f' epoch: {epoch}')
      print(avgloss/len(trains))
    return model
  except KeyboardInterrupt:
    return model
class Thiccatten(nn.Module):
  def __init__(self,k,heads):
    super().__init__()
    self.qw=nn.Linear(k,k*heads)
    self.kw=nn.Linear(k,k*heads)
    self.vw=nn.Linear(k,k*heads)
    self.fc=nn.Linear(k*heads,k)
    self.heads=heads
  def forward(self,x):
    b,t,k=x.size()
    h=self.heads
    q=self.qw(x).view(b,t,h,k)
    key=self.kw(x).view(b,t,h,k)
    v=self.vw(x).view(b,t,h,k)
    keys = key.transpose(1, 2).contiguous().view(b * h, t, k)
    queries = q.transpose(1, 2).contiguous().view(b * h, t, k)
    values = v.transpose(1, 2).contiguous().view(b * h, t, k)
    keys=keys/(k**0.25)
    queries=queries/(k**0.25)
    dot=torch.bmm(keys,queries.transpose(1,2))
    scaled_dot=torch.softmax(dot,dim=2)
    out = torch.bmm(scaled_dot, values).view(b, h, t, k)
    out=out.transpose(1,2).contiguous().view(b,t,k*h)
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
wideatten=Thiccatten
class c_transformer(nn.Module):
  def __init__(self,heads=8,depth=7,word_embed=20,max_seq=6000,mode='mean'):
    super().__init__()
    self.transformers=nn.Sequential(*[tblock(word_embed,heads) for i in range(depth)])
    self.w_embed=nn.EmbeddingBag(len(vocab)+1,word_embed,mode=mode)
    self.pos_embed=nn.Embedding(max_seq+1,word_embed)
    self.fc=nn.Linear(word_embed,1)
  def forward(self,x):
    w=torch.stack([self.w_embed(word2tensor(i).unsqueeze(0)) for i in remove_stops(x.split(' '))]).transpose(0,1).to(device)
    b,t,k=w.size()
    pos_embeddings=self.pos_embed(torch.arange(t)).expand(b,t,k)
    attended=self.transformers(pos_embeddings+w)
    classes=self.fc(attended).mean(dim=1)
    return torch.sigmoid(classes.reshape(-1))
    attended=self.transformers(pos_embeddings+w)
class native_transformer(nn.Module):
  def __init__(self,heads=8,depth=7,word_embed=20,max_seq=6000,mode='mean'):
    super().__init__()
    self.transformers=torch.jit.script(nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=word_embed,nhead=heads),depth))
    self.w_embed=nn.EmbeddingBag(len(vocab)+1,word_embed,mode=mode)
    self.pos_embed=nn.Embedding(max_seq+1,word_embed)
    self.fc=torch.jit.script(nn.Linear(word_embed,1))
    self.sig=torch.jit.script(nn.Sigmoid())
    with open('stopwords.txt','r') as df:
      for i in df:
        self.stopwords=eval(i)
  def forward(self,x):
    removed=[i for i in x.split(' ') if i not in self.stopwords]
    w=torch.stack([self.w_embed(word2tensor(i).unsqueeze(0)) for i in removed]).transpose(0,1).to(device)
    b,t,k=w.size()
    pos_embeddings=self.pos_embed(torch.arange(t)).expand(b,t,k)
    attended=self.transformers(pos_embeddings+w)
    classes=self.fc(attended).mean(dim=1)
    return self.sig(classes.reshape(-1))
