import torch
from torch import nn
from train_setup import *
import sys
torch.manual_seed(1)
if len(sys.argv)==1:
    device=torch.device('cpu')
else:
    device=torch.device(sys.argv[-1])
print(device)
#train=read_scraped_data('train.txt')
#val=read_scraped_data('val.txt')
model=c_transformer(heads=8,word_embed=30,depth=6).to(device)
already_train_better(model,train,val,lr=0.01,optim='sgd',epochs=1)
