from __future__ import division
from Data_Management.data_utils import load_partitions
import torch
from torch.utils import data
from Data_Management.dataset import Dataset
# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Parameters
params = {'batch_size': 12,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 100


partition,labels=load_partitions()


# Generators
training_set = Dataset(partition['train'], labels)
training_generator = data.DataLoader(training_set, **params)

validation_set = Dataset(partition['validation'], labels)
validation_generator = data.DataLoader(validation_set, **params)

# Loop over epochs
for epoch in range(max_epochs):
    # Training
    for local_batch, local_labels in training_generator:
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        for sample in local_batch:
        	print local_batch.shape

        # Model computations
