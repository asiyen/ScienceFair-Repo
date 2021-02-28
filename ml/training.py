import torch
from torch import nn
from train_setup import *
import sys
batch=False
torch.manual_seed(1)
if 'cuda' in sys.argv:
    device=torch.device('cuda')
else:
    device=torch.device('cpu')
print('device: ',device)
if 'batch' in sys.argv:
    btrain=batch_data(train,32)
    model=batch_transformer(heads=8,word_embed=30,depth=6).to(device)
    already_train_better(model,btrain,val,lr=0.01,optim='sgd',epochs=30)
else:
    model=c_transformer(heads=8,word_embed=30,depth=6).to(device)
    already_train_better(model,train,val,lr=0.01,optim='sgd',epochs=1)


