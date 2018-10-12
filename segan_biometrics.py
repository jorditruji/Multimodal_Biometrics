from __future__ import division
from Data_Management.data_utils import load_partitions
import torch
from torch.utils import data
from Data_Management.dataset import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from Audio.mfcc_CNN import MiniConvNet
import string
import sys
import numpy as np

# Get model nu,ber of params
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    print "Parametros: {}".format(str(pp))    
    return pp


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

def train_model(model, criterion, optimizer,scheduler, num_epochs=25):
	'''train the network'''
	since = time.time()
	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	model = model.to(device)
	model.cuda()
	max_epochs=50
	#Loop over epochs
	for epoch in range(max_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		scheduler.step()
		model.train()  # Set model to training mode
		# Training
		dataseize=0
		cont=0
		for local_batch, local_labels in training_generator:

			cont+=1
			# Transfer to GPU


			local_batch, local_labels = local_batch.to(device), local_labels.to(device)
			# Model computations

			running_loss = 0.0
			running_corrects = 0

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + shapes modification...
			local_batch=local_batch.unsqueeze_(1)
			local_batch=local_batch.type(torch.cuda.FloatTensor)
			with torch.set_grad_enabled(True):
				outputs = model(local_batch)
				_, preds = torch.max(outputs, 1)
				local_labels=torch.tensor(to_categorical(local_labels, 27), device=device).type(torch.cuda.FloatTensor)
				loss = criterion(outputs, local_labels)
				# backward + optimize only if in training phase
				loss.backward()
				optimizer.step()
				# statistics
				running_loss += loss.item() * local_batch.size(0)
				dataseize+= local_batch.size(0)
				running_corrects += torch.sum(preds == local_labels.data)
				sys.stdout.write('\r%s %s %s %s %s %s' % ('Processing training batch: ', cont, '/', training_generator.__len__(),' with loss: ', loss.item())),
				sys.stdout.flush()

		epoch_loss = running_loss / dataseize
		epoch_acc = running_corrects.double() / dataseize

		print('{} Loss: {:.4f} Acc: {:.4f}'.format("Train", epoch_loss, epoch_acc))


		        
		model.eval()   # Set model to evaluate mode

		running_loss = 0.0
		running_corrects = 0
		dataseize=0
		# Validation
		with torch.set_grad_enabled(False):
			for local_batch, local_labels in validation_generator:
				# Transfer to GPU
				# forward + shapes modification...
				local_batch=local_batch.unsqueeze_(1)
				local_batch=local_batch.type(torch.cuda.FloatTensor)
				local_batch, local_labels = local_batch.to(device), local_labels.to(device)
				outputs = model(local_batch)
				_, preds = torch.max(outputs, 1)
				loss = criterion(outputs, local_labels)


				# statistics
				running_loss += loss.item() * local_batch.size(0)
				running_corrects += torch.sum(preds == local_labels.data)
				dataseize+= local_batch.size(0)


			epoch_loss = running_loss / dataseize
			epoch_acc = running_corrects.double() / dataseize

			print('Val Loss: {:.4f} Acc: {:.4f}'.format( epoch_loss, epoch_acc))

			# deep copy the model
			if  epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))
	# load best model weights
	model.load_state_dict(best_model_wts)
	return model.cuda()

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print device

# Parameters
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}


partition,labels=load_partitions()



# Generators
training_set = Dataset(partition['train'], labels)
training_generator = data.DataLoader(training_set, **params)

validation_set = Dataset(partition['validation'], labels)
validation_generator = data.DataLoader(validation_set, **params)



model = MiniConvNet(num_classes=27).to(device)


model_ft = model.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model_ft = train_model(model_ft, criterion, optimizer_ft,exp_lr_scheduler, num_epochs=50)




'''

# Loop over epochs
for epoch in range(max_epochs):
    # Training
    for local_batch, local_labels in training_generator:
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        for sample in local_batch:
        	print local_batch.shape

        # Model computations
'''