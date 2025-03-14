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
w1 = c*u1
w2 = c*(p*u1+np.sqrt((1-p**2))*u2)
w3 = c*(0.8*u1+np.sqrt((1-0.8**2))*u2)
w4 = c*(0.7*u1+np.sqrt((1-0.7**2))*u2)
w5 = c*(0.6*u1+np.sqrt((1-0.6**2))*u2)
w6 = c*(0.5*u1+np.sqrt((1-0.5**2))*u2)
w7 = c*(0.4*u1+np.sqrt((1-0.4**2))*u2)
w8 = c*(0.3*u1+np.sqrt((1-0.3**2))*u2)
w9 = c*(0.2*u1+np.sqrt((1-0.2**2))*u2)

X = np.random.normal(0, 1, (N, M))
eps = np.random.normal(0, 0.01, 9)  # 9 different noise terms

# Generate 9 different outputs
Y1 = np.matmul(X, w1) + np.sin(np.matmul(X, w1))+eps[0]
Y2 = np.matmul(X, w2) + np.sin(np.matmul(X, w2))+eps[1]
Y3 = np.matmul(X, w3) + np.sin(np.matmul(X, w3))+eps[2]
Y4 = np.matmul(X, w4) + np.sin(np.matmul(X, w4))+eps[3]
Y5 = np.matmul(X, w5) + np.sin(np.matmul(X, w5))+eps[4]
Y6 = np.matmul(X, w6) + np.sin(np.matmul(X, w6))+eps[5]
Y7 = np.matmul(X, w7) + np.sin(np.matmul(X, w7))+eps[6]
Y8 = np.matmul(X, w8) + np.sin(np.matmul(X, w8))+eps[7]
Y9 = np.matmul(X, w9) + np.sin(np.matmul(X, w9))+eps[8]

split = list(np.random.permutation(N))

X_train = X[split[0:8000],:]
Y1_train = Y1[split[0:8000]]
Y2_train = Y2[split[0:8000]]
Y3_train = Y3[split[0:8000]]
Y4_train = Y4[split[0:8000]]
Y5_train = Y5[split[0:8000]]
Y6_train = Y6[split[0:8000]]
Y7_train = Y7[split[0:8000]]
Y8_train = Y8[split[0:8000]]
Y9_train = Y9[split[0:8000]]

X_valid = X[8000:9000,:]
Y1_valid = Y1[8000:9000]
Y2_valid = Y2[8000:9000]
Y3_valid = Y3[8000:9000]
Y4_valid = Y4[8000:9000]
Y5_valid = Y5[8000:9000]
Y6_valid = Y6[8000:9000]
Y7_valid = Y7[8000:9000]
Y8_valid = Y8[8000:9000]
Y9_valid = Y9[8000:9000]

X_test = X[9000:10000,:]
Y1_test = Y1[9000:10000]
Y2_test = Y2[9000:10000]
Y3_test = Y3[9000:10000]
Y4_test = Y4[9000:10000]
Y5_test = Y5[9000:10000]
Y6_test = Y6[9000:10000]
Y7_test = Y7[9000:10000]
Y8_test = Y8[9000:10000]
Y9_test = Y9[9000:10000]

print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)
print(Y1_train.shape)
print(Y2_train.shape)
print(Y1_valid.shape)
print(Y2_valid.shape)
print(Y1_test.shape)
print(Y2_test.shape)

X_train = torch.from_numpy(X_train).float().to(device)
Y1_train = torch.tensor(Y1_train).float().to(device)
Y2_train = torch.tensor(Y2_train).float().to(device)
Y3_train = torch.tensor(Y3_train).float().to(device)
Y4_train = torch.tensor(Y4_train).float().to(device)
Y5_train = torch.tensor(Y5_train).float().to(device)
Y6_train = torch.tensor(Y6_train).float().to(device)
Y7_train = torch.tensor(Y7_train).float().to(device)
Y8_train = torch.tensor(Y8_train).float().to(device)
Y9_train = torch.tensor(Y9_train).float().to(device)

X_valid = torch.from_numpy(X_valid).float().to(device)
Y1_valid = torch.tensor(Y1_valid).float().to(device)
Y2_valid = torch.tensor(Y2_valid).float().to(device)
Y3_valid = torch.tensor(Y3_valid).float().to(device)
Y4_valid = torch.tensor(Y4_valid).float().to(device)
Y5_valid = torch.tensor(Y5_valid).float().to(device)
Y6_valid = torch.tensor(Y6_valid).float().to(device)
Y7_valid = torch.tensor(Y7_valid).float().to(device)
Y8_valid = torch.tensor(Y8_valid).float().to(device)
Y9_valid = torch.tensor(Y9_valid).float().to(device)

X_test = torch.from_numpy(X_test).float().to(device)
Y1_test = torch.tensor(Y1_test).float().to(device)
Y2_test = torch.tensor(Y2_test).float().to(device)
Y3_test = torch.tensor(Y3_test).float().to(device)
Y4_test = torch.tensor(Y4_test).float().to(device)
Y5_test = torch.tensor(Y5_test).float().to(device)
Y6_test = torch.tensor(Y6_test).float().to(device)
Y7_test = torch.tensor(Y7_test).float().to(device)
Y8_test = torch.tensor(Y8_test).float().to(device)
Y9_test = torch.tensor(Y9_test).float().to(device)

print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)
print(Y1_train.shape)
print(Y2_train.shape)
print(Y1_valid.shape)
print(Y2_valid.shape)
print(Y1_test.shape)
print(Y2_test.shape)


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
cost1tr, cost2tr, cost3tr, cost4tr, cost5tr, cost6tr, cost7tr, cost8tr, cost9tr = [], [], [], [], [], [], [], [], []
cost1D, cost2D, cost3D, cost4D, cost5D, cost6D, cost7D, cost8D, cost9D = [], [], [], [], [], [], [], [], []
cost1ts, cost2ts, cost3ts, cost4ts, cost5ts, cost6ts, cost7ts, cost8ts, cost9ts = [], [], [], [], [], [], [], [], []
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
        
        # Create 9 towers for 9 different tasks
        self.tower1 = nn.Sequential(
            nn.Linear(shared_layer_size, tower_h1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(tower_h1, tower_h2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(tower_h2, output_size)
        )
        self.tower2 = nn.Sequential(
            nn.Linear(shared_layer_size, tower_h1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(tower_h1, tower_h2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(tower_h2, output_size)
        )
        self.tower3 = nn.Sequential(
            nn.Linear(shared_layer_size, tower_h1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(tower_h1, tower_h2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(tower_h2, output_size)
        )
        self.tower4 = nn.Sequential(
            nn.Linear(shared_layer_size, tower_h1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(tower_h1, tower_h2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(tower_h2, output_size)
        )
        self.tower5 = nn.Sequential(
            nn.Linear(shared_layer_size, tower_h1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(tower_h1, tower_h2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(tower_h2, output_size)
        )
        self.tower6 = nn.Sequential(
            nn.Linear(shared_layer_size, tower_h1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(tower_h1, tower_h2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(tower_h2, output_size)
        )
        self.tower7 = nn.Sequential(
            nn.Linear(shared_layer_size, tower_h1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(tower_h1, tower_h2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(tower_h2, output_size)
        )
        self.tower8 = nn.Sequential(
            nn.Linear(shared_layer_size, tower_h1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(tower_h1, tower_h2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(tower_h2, output_size)
        )
        self.tower9 = nn.Sequential(
            nn.Linear(shared_layer_size, tower_h1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(tower_h1, tower_h2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(tower_h2, output_size)
        )

    def forward(self, x):
        h_shared = self.sharedlayer(x)
        out1 = self.tower1(h_shared)
        out2 = self.tower2(h_shared)
        out3 = self.tower3(h_shared)
        out4 = self.tower4(h_shared)
        out5 = self.tower5(h_shared)
        out6 = self.tower6(h_shared)
        out7 = self.tower7(h_shared)
        out8 = self.tower8(h_shared)
        out9 = self.tower9(h_shared)
        return out1, out2, out3, out4, out5, out6, out7, out8, out9

def random_mini_batches(XE, R1E, R2E, R3E, R4E, R5E, R6E, R7E, R8E, R9E, mini_batch_size=10, seed=42): 
    # Creating the mini-batches
    np.random.seed(seed)            
    m = XE.shape[0]                  
    mini_batches = []
    permutation = list(np.random.permutation(m))
    shuffled_XE = XE[permutation,:]
    shuffled_X1R = R1E[permutation]
    shuffled_X2R = R2E[permutation]
    shuffled_X3R = R3E[permutation]
    shuffled_X4R = R4E[permutation]
    shuffled_X5R = R5E[permutation]
    shuffled_X6R = R6E[permutation]
    shuffled_X7R = R7E[permutation]
    shuffled_X8R = R8E[permutation]
    shuffled_X9R = R9E[permutation]
    
    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, int(num_complete_minibatches)):
        mini_batch_XE = shuffled_XE[k * mini_batch_size : (k+1) * mini_batch_size, :]
        mini_batch_X1R = shuffled_X1R[k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch_X2R = shuffled_X2R[k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch_X3R = shuffled_X3R[k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch_X4R = shuffled_X4R[k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch_X5R = shuffled_X5R[k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch_X6R = shuffled_X6R[k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch_X7R = shuffled_X7R[k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch_X8R = shuffled_X8R[k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch_X9R = shuffled_X9R[k * mini_batch_size : (k+1) * mini_batch_size]
        
        mini_batch = (mini_batch_XE, mini_batch_X1R, mini_batch_X2R, mini_batch_X3R, 
                     mini_batch_X4R, mini_batch_X5R, mini_batch_X6R, mini_batch_X7R, 
                     mini_batch_X8R, mini_batch_X9R)
        mini_batches.append(mini_batch)
    
    Lower = int(num_complete_minibatches * mini_batch_size)
    Upper = int(m - (mini_batch_size * math.floor(m/mini_batch_size)))
    if m % mini_batch_size != 0:
        mini_batch_XE = shuffled_XE[Lower : Lower + Upper, :]
        mini_batch_X1R = shuffled_X1R[Lower : Lower + Upper]
        mini_batch_X2R = shuffled_X2R[Lower : Lower + Upper]
        mini_batch_X3R = shuffled_X3R[Lower : Lower + Upper]
        mini_batch_X4R = shuffled_X4R[Lower : Lower + Upper]
        mini_batch_X5R = shuffled_X5R[Lower : Lower + Upper]
        mini_batch_X6R = shuffled_X6R[Lower : Lower + Upper]
        mini_batch_X7R = shuffled_X7R[Lower : Lower + Upper]
        mini_batch_X8R = shuffled_X8R[Lower : Lower + Upper]
        mini_batch_X9R = shuffled_X9R[Lower : Lower + Upper]
        
        mini_batch = (mini_batch_XE, mini_batch_X1R, mini_batch_X2R, mini_batch_X3R, 
                     mini_batch_X4R, mini_batch_X5R, mini_batch_X6R, mini_batch_X7R, 
                     mini_batch_X8R, mini_batch_X9R)
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
    epoch_cost1, epoch_cost2, epoch_cost3 = 0, 0, 0
    epoch_cost4, epoch_cost5, epoch_cost6 = 0, 0, 0
    epoch_cost7, epoch_cost8, epoch_cost9 = 0, 0, 0
    
    num_minibatches = int(input_size / mb_size) 
    minibatches = random_mini_batches(X_train, Y1_train, Y2_train, Y3_train, Y4_train, 
                                      Y5_train, Y6_train, Y7_train, Y8_train, Y9_train, mb_size)
    
    for minibatch in minibatches:
        XE, YE1, YE2, YE3, YE4, YE5, YE6, YE7, YE8, YE9 = minibatch 
        
        XE = XE.to(device)  
        YE1, YE2, YE3 = YE1.to(device), YE2.to(device), YE3.to(device)
        YE4, YE5, YE6 = YE4.to(device), YE5.to(device), YE6.to(device)
        YE7, YE8, YE9 = YE7.to(device), YE8.to(device), YE9.to(device)
        
        Yhat1, Yhat2, Yhat3, Yhat4, Yhat5, Yhat6, Yhat7, Yhat8, Yhat9 = MTL(XE)
        
        l1 = loss_func(Yhat1, YE1.view(-1,1))    
        l2 = loss_func(Yhat2, YE2.view(-1,1))
        l3 = loss_func(Yhat3, YE3.view(-1,1))
        l4 = loss_func(Yhat4, YE4.view(-1,1))
        l5 = loss_func(Yhat5, YE5.view(-1,1))
        l6 = loss_func(Yhat6, YE6.view(-1,1))
        l7 = loss_func(Yhat7, YE7.view(-1,1))
        l8 = loss_func(Yhat8, YE8.view(-1,1))
        l9 = loss_func(Yhat9, YE9.view(-1,1))
        
        # Total loss is the average of all task losses
        loss = (l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8 + l9) / 9
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_cost = epoch_cost + (loss / num_minibatches)
        epoch_cost1 = epoch_cost1 + (l1 / num_minibatches)
        epoch_cost2 = epoch_cost2 + (l2 / num_minibatches)
        epoch_cost3 = epoch_cost3 + (l3 / num_minibatches)
        epoch_cost4 = epoch_cost4 + (l4 / num_minibatches)
        epoch_cost5 = epoch_cost5 + (l5 / num_minibatches)
        epoch_cost6 = epoch_cost6 + (l6 / num_minibatches)
        epoch_cost7 = epoch_cost7 + (l7 / num_minibatches)
        epoch_cost8 = epoch_cost8 + (l8 / num_minibatches)
        epoch_cost9 = epoch_cost9 + (l9 / num_minibatches)
        
    costtr.append(torch.mean(epoch_cost))
    cost1tr.append(torch.mean(epoch_cost1))
    cost2tr.append(torch.mean(epoch_cost2))
    cost3tr.append(torch.mean(epoch_cost3))
    cost4tr.append(torch.mean(epoch_cost4))
    cost5tr.append(torch.mean(epoch_cost5))
    cost6tr.append(torch.mean(epoch_cost6))
    cost7tr.append(torch.mean(epoch_cost7))
    cost8tr.append(torch.mean(epoch_cost8))
    cost9tr.append(torch.mean(epoch_cost9))
    
    with torch.no_grad():
        X_valid = X_valid.to(device)
        Y1_valid, Y2_valid, Y3_valid = Y1_valid.to(device), Y2_valid.to(device), Y3_valid.to(device)
        Y4_valid, Y5_valid, Y6_valid = Y4_valid.to(device), Y5_valid.to(device), Y6_valid.to(device)
        Y7_valid, Y8_valid, Y9_valid = Y7_valid.to(device), Y8_valid.to(device), Y9_valid.to(device)
        
        Yhat1D, Yhat2D, Yhat3D, Yhat4D, Yhat5D, Yhat6D, Yhat7D, Yhat8D, Yhat9D = MTL(X_valid)
        
        l1D = loss_func(Yhat1D, Y1_valid.view(-1,1))
        l2D = loss_func(Yhat2D, Y2_valid.view(-1,1))
        l3D = loss_func(Yhat3D, Y3_valid.view(-1,1))
        l4D = loss_func(Yhat4D, Y4_valid.view(-1,1))
        l5D = loss_func(Yhat5D, Y5_valid.view(-1,1))
        l6D = loss_func(Yhat6D, Y6_valid.view(-1,1))
        l7D = loss_func(Yhat7D, Y7_valid.view(-1,1))
        l8D = loss_func(Yhat8D, Y8_valid.view(-1,1))
        l9D = loss_func(Yhat9D, Y9_valid.view(-1,1))
        
        cost1D.append(l1D)
        cost2D.append(l2D)
        cost3D.append(l3D)
        cost4D.append(l4D)
        cost5D.append(l5D)
        cost6D.append(l6D)
        cost7D.append(l7D)
        cost8D.append(l8D)
        cost9D.append(l9D)
        
        costD.append((l1D + l2D + l3D + l4D + l5D + l6D + l7D + l8D + l9D) / 9)
        print('Iter-{}; Total loss: {:.4}'.format(it, loss.item()))
    
# Plot total cost
plt.plot([t.cpu().detach().numpy() for t in costtr], '-r', [t.cpu().detach().numpy() for t in costD], '-b')
plt.ylabel('total cost')
plt.xlabel('iterations (per tens)')
plt.savefig('total_cost.png')
plt.close()
plt.show()

# Plot individual task costs
for i, (cost_tr, cost_val, task_num) in enumerate([
    (cost1tr, cost1D, 1), (cost2tr, cost2D, 2), (cost3tr, cost3D, 3),
    (cost4tr, cost4D, 4), (cost5tr, cost5D, 5), (cost6tr, cost6D, 6),
    (cost7tr, cost7D, 7), (cost8tr, cost8D, 8), (cost9tr, cost9D, 9)
]):
    plt.plot([t.cpu().detach().numpy() for t in cost_tr], '-r', 
             [t.cpu().detach().numpy() for t in cost_val], '-b')
    plt.ylabel(f'task {task_num} cost')
    plt.xlabel('iterations (per tens)')
    plt.savefig(f'figs/task{task_num}_cost.png')
    plt.close()
    plt.show()
# In[ ]:





# I