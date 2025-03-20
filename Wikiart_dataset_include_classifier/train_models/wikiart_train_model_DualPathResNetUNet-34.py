import sys
import os

# 将上级目录添加到 sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import copy
from d2l import torch as d2l
from models.DualPathResNetUNet import DualPathResNetUNet
from dataloader import dataloader_test,dataloader_train,dataloader_val,batch_size

USE_GPU = True
dtype = torch.float32
print_every = 100

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def check_accuracy(loader, model):
    if loader.dataset.mode == 'val':
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval()   # set model to evaluation mode
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _,preds = scores.max(1)
            num_correct += (preds==y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 *acc ))
        return acc

def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)

def calculate_accuracy(scores, y):
    _, preds = scores.max(1)
    num_correct = (preds == y).sum()
    num_samples = preds.size(0)
    acc = float(num_correct) / num_samples
    return acc

def train_model(model, optimizer, epochs=1, scheduler=None,dataloader_train=dataloader_train,dataloader_val=dataloader_val):
    '''
    Parameters:
    - model: A Pytorch Module giving the model to train.
    - optimizer: An optimizer object we will use to train the model
    - epochs: A Python integer giving the number of epochs to train
    Returns: best model
    '''
    animator = d2l.Animator(xlabel='epoch', xlim=[1, epochs],
                            legend=['train loss', 'train acc', 'val acc'])
    best_model_wts = None
    best_acc = 0.0
    model = model.to(device=device) # move the model parameters to CPU/GPU
    for e in range(epochs):
        
        for t,(x,y) in enumerate(dataloader_train):
            model.train()   # set model to training mode
            x = x.to(device, dtype=dtype)
            y = y.to(device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)
            metric = d2l.Accumulator(3)
            accuracy = calculate_accuracy(scores, y)
            metric.add(float(loss), accuracy, y.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            break

        print('Epoch %d, loss=%.4f, train_accuracy=%.4f' % (e, loss.item(),accuracy))
        acc = check_accuracy(dataloader_val, model)
        animator.add(e + 1, (metric[0], metric[1]) + (acc,))
        if acc > best_acc:
            best_model_wts = copy.deepcopy(model.state_dict())
            best_acc = acc
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(acc)
            else:
                scheduler.step()
    print('best_acc:',best_acc)
    model.load_state_dict(best_model_wts)
    return model
learning_rate = 1e-1
net = DualPathResNetUNet(num_classes=11)
epochs = 5
#optimizer_resnet = optim.SGD(resnet.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
weight_decay = 5e-4
#optimizer_net=optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
optimizer_net = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
#scheduler=lr_scheduler.ReduceLROnPlateau(optimizer_net, mode='max', factor=0.5, patience=5,)
scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer_net, T_max=50)
def which_model():
    return 'DualPathResNetUNet-34'
#optimizer_resnet = optim.Adam(resnet.parameters(), lr=learning_rate,weight_decay=weight_decay)
#scheduler = lr_scheduler.StepLR(optimizer_resnet, step_size=15,gamma=0.1)
best_net = train_model(net, optimizer_net,epochs,scheduler)
save_checkpoint({
    'epoch': epochs,
    'model_state_dict': best_net.state_dict(),
    'optimizer_state_dict': optimizer_net.state_dict(),
}, filename=f'checkpoint/{which_model()}_lr{learning_rate}_sgd_batchsize{batch_size}_epochs{epochs}_weight_decay{weight_decay}.pth')

test_acc=check_accuracy(dataloader_test, best_net)

d2l.plt.savefig(f'figs/{which_model()}_lr{learning_rate}_sgd_batchsize{batch_size}_epochs{epochs}_weight_decay{weight_decay}.png')

def save_test_accuracy(accuracy, filename):
    with open(filename, 'a') as f:
        f.write(f"{which_model()}__lr{learning_rate}_sgd_batchsize{batch_size}_epochs{epochs}_weight_decay{weight_decay}:test_acc{accuracy}"+"\n")
        
save_test_accuracy(test_acc, f'acc_history/{which_model()}_test_accuracy_weight_decay{weight_decay}.txt')