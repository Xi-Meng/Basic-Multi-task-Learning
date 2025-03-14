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


# Check if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    multi_gpu = True
else:
    multi_gpu = False
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Main device: {device}")

# Print available GPUs
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    torch.cuda.init()
    torch.cuda.empty_cache()

# In[12]:


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

N = 10000
M = 100
c = 0.5
k = np.random.randn(M)
u1 = np.random.randn(M)
u1 -= u1.dot(k) * k / np.linalg.norm(k)**2
u1 /= np.linalg.norm(u1) 
k /= np.linalg.norm(k) 
u2 = k

# Load data from data_file
data_file = "/home/mengyuex/scratch/workspace/mtl/data/gh100_nvpdm_aarch_combined.csv"
data = np.loadtxt(data_file, delimiter=',')

# Extract features and targets
X = data[:, :-8]  # All columns except the last 8
Y_values = [data[:, -8+i] for i in range(8)]  # Last 8 columns as separate outputs

N, M = X.shape  # N samples, M features
print(f"Loaded data with {N} samples and {M} features")

# Create correlation values for reference (assuming decreasing correlation)
correlations = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]

# Create random split indices
split = list(np.random.permutation(N))

# Split data into train, validation, and test sets (80%, 10%, 10%)
train_size = int(0.8 * N)
valid_size = int(0.1 * N)

X_train = X[split[0:train_size], :]
X_valid = X[split[train_size:train_size+valid_size], :]
X_test = X[split[train_size+valid_size:], :]

# Create train, validation, and test sets for all 8 tasks
Y_train = [Y[split[0:train_size]] for Y in Y_values]
Y_valid = [Y[split[train_size:train_size+valid_size]] for Y in Y_values]
Y_test = [Y[split[train_size+valid_size:]] for Y in Y_values]

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

# Create cost lists for all 8 tasks
cost_tr_tasks = [[] for _ in range(8)]
cost_val_tasks = [[] for _ in range(8)]
cost_ts_tasks = [[] for _ in range(8)]
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
        
        # Create 8 towers for 8 different tasks using a list comprehension
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_layer_size, tower_h1),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(tower_h1, tower_h2),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(tower_h2, output_size)
            ) for _ in range(8)
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


# Create model
MTL = MTLnet()

# Move model to GPU and enable multi-GPU training if available
if multi_gpu:
    print("Using DataParallel to utilize multiple GPUs")
    MTL = nn.DataParallel(MTL)
MTL.to(device)

# Create optimizer and loss function
optimizer = torch.optim.Adam(MTL.parameters(), lr=LR)
loss_func = nn.MSELoss()


# In[15]:


# Create directory for figures if it doesn't exist
os.makedirs('figs', exist_ok=True)

for it in range(epoch):
    epoch_cost = 0
    epoch_costs = [0] * 8
    
    num_minibatches = int(input_size / mb_size) 
    minibatches = random_mini_batches(X_train, Y_train_tensors, mb_size, seed=it)
    
    MTL.train()  # Set model to training mode
    for minibatch in minibatches:
        XE, YE_list = minibatch 
        
        XE = XE.to(device)
        YE_list = [YE.to(device) for YE in YE_list]
        
        Yhats = MTL(XE)
        
        losses = [loss_func(Yhats[i], YE_list[i].view(-1,1)) for i in range(8)]
        
        # Total loss is the average of all task losses
        loss = sum(losses) / 8
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_cost = epoch_cost + (loss / num_minibatches)
        for i in range(8):
            epoch_costs[i] = epoch_costs[i] + (losses[i] / num_minibatches)
        
    costtr.append(torch.mean(epoch_cost))
    for i in range(8):
        cost_tr_tasks[i].append(torch.mean(epoch_costs[i]))
    
    MTL.eval()  # Set model to evaluation mode
    with torch.no_grad():
        X_valid = X_valid.to(device)
        Y_valid_list = [Y.to(device) for Y in Y_valid_tensors]
        
        Yhats_valid = MTL(X_valid)
        
        val_losses = [loss_func(Yhats_valid[i], Y_valid_list[i].view(-1,1)) for i in range(8)]
        
        for i in range(8):
            cost_val_tasks[i].append(val_losses[i])
        
        costD.append(sum(val_losses) / 8)
        print('Iter-{}; Total loss: {:.4}'.format(it, loss.item()))
    
# Plot total cost
plt.plot([t.cpu().detach().numpy() for t in costtr], '-r', [t.cpu().detach().numpy() for t in costD], '-b')
plt.ylabel('total cost')
plt.xlabel('iterations (per tens)')
plt.savefig('figs/total_cost.png')
plt.close()
plt.show()

def plot_task_costs(cost_tr_tasks, cost_val_tasks, save_path='figs/all_tasks_cost.png'):
    """
    Plots individual task costs in a single figure with multiple subplots.
    
    Args:
        cost_tr_tasks: List of lists containing training costs for each task
        cost_val_tasks: List of lists containing validation costs for each task
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i in range(8):
        axes[i].plot([t.cpu().detach().numpy() for t in cost_tr_tasks[i]], '-r', 
                     [t.cpu().detach().numpy() for t in cost_val_tasks[i]], '-b')
        axes[i].set_ylabel(f'task {i+1} cost')
        axes[i].set_xlabel('iterations (per tens)')
        axes[i].set_title(f'Task {i+1}')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    plt.show()

# Call the function to plot task costs
plot_task_costs(cost_tr_tasks, cost_val_tasks)



# Calculate and plot final accuracy for each task on validation dataset
def evaluate_and_visualize_model(model, X_valid, Y_valid_tensors, correlations, device):
    """
    Evaluates the model on validation data and creates visualization plots.
    
    Args:
        model: The trained MTL model
        X_valid: Validation input data tensor
        Y_valid_tensors: List of validation target tensors for each task
        correlations: List of correlation values for each task
        device: Device to run evaluation on (cuda or cpu)
    
    Returns:
        dict: Dictionary containing evaluation metrics for each task
    """
    model.eval()
    results = {}
    
    with torch.no_grad():
        # Get predictions for validation set
        Yhats_valid = model(X_valid)
        
        # Create a figure with 8 subplots for each task
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i in range(8):
            # Get predictions and true values
            y_pred = Yhats_valid[i].cpu().numpy().flatten()
            y_true = Y_valid_tensors[i].cpu().numpy().flatten()
            
            # Calculate metrics
            mse = metrics.mean_squared_error(y_true, y_pred)
            r2 = metrics.r2_score(y_true, y_pred)
            mae = metrics.mean_absolute_error(y_true, y_pred)
            
            # Store results
            results[f'task_{i+1}'] = {
                'mse': mse,
                'r2': r2,
                'mae': mae,
                'correlation': correlations[i]
            }
            
            # Plot true vs predicted values
            axes[i].scatter(y_true, y_pred, alpha=0.5)
            axes[i].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
            
            # Add regression line
            z = np.polyfit(y_true, y_pred, 1)
            p = np.poly1d(z)
            axes[i].plot(y_true, p(y_true), "b--", alpha=0.7)
            
            axes[i].set_xlabel('True Values')
            axes[i].set_ylabel('Predictions')
            axes[i].set_title(f'Task {i+1} (corr={correlations[i]:.2f})\n'
                             f'MSE: {mse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}')
            
            # Add text with metrics
            axes[i].text(0.05, 0.95, f'MSE: {mse:.4f}\nR²: {r2:.4f}\nMAE: {mae:.4f}', 
                     transform=axes[i].transAxes, 
                     bbox=dict(facecolor='white', alpha=0.8))
            
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('figs/all_tasks_validation_accuracy.png')
        plt.close()
        
        print("Final validation metrics:")
        for i in range(8):
            task_results = results[f'task_{i+1}']
            print(f"Task {i+1} (correlation={task_results['correlation']:.2f}): "
                  f"MSE={task_results['mse']:.4f}, R²={task_results['r2']:.4f}, MAE={task_results['mae']:.4f}")
    
    return results

# Call the function
evaluation_results = evaluate_and_visualize_model(MTL, X_valid, Y_valid_tensors, correlations, device)
# %%
