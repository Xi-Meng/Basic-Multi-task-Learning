#!/usr/bin/env python
# coding: utf-8

# In[11]:


# import sys
# sys.path.append('/home/scratch.mengyuex_gpu/pip')
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
import pandas as pd
import math
import sklearn.preprocessing as sk
from tensorboardX import SummaryWriter
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Add this right after the imports, before any other code
if torch.cuda.is_available():
    torch.cuda.init()
    torch.cuda.empty_cache()

# In[12]:


seed = 42
random.seed(seed)
torch.cuda.manual_seed_all(seed)

N = 10000
M = 100
c = 0.5
p = 0.9
k = np.random.randn(M)
u1 = np.random.randn(M)
u1 -= u1.dot(k) * k / np.linalg.norm(k)**2
u1 /= np.linalg.norm(u1) 
k /= np.linalg.norm(k) 
u2 = k

# Create 9 different weight vectors with varying correlations
correlations = [1.0, p, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
weights = []
for corr in correlations:
    if corr == 1.0:  # Special case for w1
        weights.append(c * u1)
    else:
        weights.append(c * (corr * u1 + np.sqrt(1 - corr**2) * u2))

X = np.random.normal(0, 1, (N, M))
eps = np.random.normal(0, 0.01, 9)  # 9 different noise terms

# Generate 9 different outputs
Y_values = []
for i in range(9):
    w = weights[i]
    Y = np.matmul(X, w) + np.sin(np.matmul(X, w)) + eps[i]
    Y_values.append(Y)

split = list(np.random.permutation(N))

# Split data into train, validation, and test sets
X_train = X[split[0:8000],:]
X_valid = X[8000:9000,:]
X_test = X[9000:10000,:]

# Create train, validation, and test sets for all 9 tasks
Y_train = [Y[split[0:8000]] for Y in Y_values]
Y_valid = [Y[8000:9000] for Y in Y_values]
Y_test = [Y[9000:10000] for Y in Y_values]

print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)
print(Y_train[0].shape)
print(Y_train[1].shape)
print(Y_valid[0].shape)
print(Y_valid[1].shape)
print(Y_test[0].shape)
print(Y_test[1].shape)

# Convert numpy arrays to PyTorch tensors and move to device
X_train = torch.from_numpy(X_train).float().to(device)
X_valid = torch.from_numpy(X_valid).float().to(device)
X_test = torch.from_numpy(X_test).float().to(device)

# Convert Y values to tensors and move to device
Y_train_tensors = [torch.tensor(y).float().to(device) for y in Y_train]
Y_valid_tensors = [torch.tensor(y).float().to(device) for y in Y_valid]
Y_test_tensors = [torch.tensor(y).float().to(device) for y in Y_test]

print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)
print(Y_train_tensors[0].shape)
print(Y_train_tensors[1].shape)
print(Y_valid_tensors[0].shape)
print(Y_valid_tensors[1].shape)
print(Y_test_tensors[0].shape)
print(Y_test_tensors[1].shape)


# In[13]:


input_size, feature_size = X.shape
shared_layer_size = 64
tower_h1 = 32
tower_h2 = 16
output_size = 1
LR = 0.001
epoch = 50
mb_size = 100

# Create cost lists for all 9 tasks
cost_tr_tasks = [[] for _ in range(9)]
cost_val_tasks = [[] for _ in range(9)]
cost_ts_tasks = [[] for _ in range(9)]
costtr = []
costD = []
costts = []

class MTLnet(nn.Module):
    def __init__(self):
        super(MTLnet, self).__init__()

        self.sharedlayer = nn.Sequential(
            nn.Linear(feature_size, shared_layer_size),
            nn.ReLU(),
            nn.Dropout()
        )
        
        # Create 9 towers for 9 different tasks using a list comprehension
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_layer_size, tower_h1),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(tower_h1, tower_h2),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(tower_h2, output_size)
            ) for _ in range(9)
        ])

    def forward(self, x):
        h_shared = self.sharedlayer(x)
        outputs = [tower(h_shared) for tower in self.towers]
        return tuple(outputs)

def random_mini_batches(XE, YE_list, mini_batch_size=10, seed=42): 
    # Creating the mini-batches
    np.random.seed(seed)            
    m = XE.shape[0]                  
    mini_batches = []
    permutation = list(np.random.permutation(m))
    shuffled_XE = XE[permutation,:]
    shuffled_YE = [YE[permutation] for YE in YE_list]
    
    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, int(num_complete_minibatches)):
        mini_batch_XE = shuffled_XE[k * mini_batch_size : (k+1) * mini_batch_size, :]
        mini_batch_YE = [YE[k * mini_batch_size : (k+1) * mini_batch_size] for YE in shuffled_YE]
        
        mini_batch = (mini_batch_XE, mini_batch_YE)
        mini_batches.append(mini_batch)
    
    Lower = int(num_complete_minibatches * mini_batch_size)
    Upper = int(m - (mini_batch_size * math.floor(m/mini_batch_size)))
    if m % mini_batch_size != 0:
        mini_batch_XE = shuffled_XE[Lower : Lower + Upper, :]
        mini_batch_YE = [YE[Lower : Lower + Upper] for YE in shuffled_YE]
        
        mini_batch = (mini_batch_XE, mini_batch_YE)
        mini_batches.append(mini_batch)
    
    return mini_batches



MTL = MTLnet()
MTL = nn.DataParallel(MTL)
MTL.to(device)
optimizer = torch.optim.Adam(MTL.parameters(), lr=LR)
loss_func = nn.MSELoss()


# In[15]:


for it in range(epoch):
    epoch_cost = 0
    epoch_costs = [0] * 9
    
    num_minibatches = int(input_size / mb_size) 
    minibatches = random_mini_batches(X_train, Y_train_tensors, mb_size)
    
    for minibatch in minibatches:
        XE, YE_list = minibatch 
        
        XE = XE.to(device)
        YE_list = [YE.to(device) for YE in YE_list]
        
        Yhats = MTL(XE)
        
        losses = [loss_func(Yhats[i], YE_list[i].view(-1,1)) for i in range(9)]
        
        # Total loss is the average of all task losses
        loss = sum(losses) / 9
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_cost = epoch_cost + (loss / num_minibatches)
        for i in range(9):
            epoch_costs[i] = epoch_costs[i] + (losses[i] / num_minibatches)
        
    costtr.append(torch.mean(epoch_cost))
    for i in range(9):
        cost_tr_tasks[i].append(torch.mean(epoch_costs[i]))
    
    with torch.no_grad():
        X_valid = X_valid.to(device)
        Y_valid_list = [Y.to(device) for Y in Y_valid_tensors]
        
        Yhats_valid = MTL(X_valid)
        
        val_losses = [loss_func(Yhats_valid[i], Y_valid_list[i].view(-1,1)) for i in range(9)]
        
        for i in range(9):
            cost_val_tasks[i].append(val_losses[i])
        
        costD.append(sum(val_losses) / 9)
        print('Iter-{}; Total loss: {:.4}'.format(it, loss.item()))
    
# Plot total cost
plt.plot([t.cpu().detach().numpy() for t in costtr], '-r', [t.cpu().detach().numpy() for t in costD], '-b')
plt.ylabel('total cost')
plt.xlabel('iterations (per tens)')
plt.savefig('figs/total_cost.png')
plt.close()
plt.show()

# Plot individual task costs
for i in range(9):
    plt.plot([t.cpu().detach().numpy() for t in cost_tr_tasks[i]], '-r', 
             [t.cpu().detach().numpy() for t in cost_val_tasks[i]], '-b')
    plt.ylabel(f'task {i+1} cost')
    plt.xlabel('iterations (per tens)')
    plt.savefig(f'figs/task{i+1}_cost.png')
    plt.close()
    plt.show()
# In[ ]:





# I