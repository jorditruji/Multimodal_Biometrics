from __future__ import division
from Data_Management.data_utils import load_partitions
import torch
from torch.utils import data
from Data_Management.dataset import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from torchvision import models
import copy
from Audio.mfcc_CNN import MiniConvNet, ConvNet, MiniConvNet2, MiniVGG, DeepSpeakerModel
from Audio.fran import Discriminator
import string
import sys
import numpy as np
from torch.autograd import Variable
import time

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
	max_epochs=50
	model.train()
	#Loop over epochs
	for epoch in range(max_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		# Training
		dataseize=0
		cont=0
		running_loss = 0.0
		running_corrects = 0
		for local_batch, local_labels in training_generator:
			cont+=1
			# Transfer to GPU
			local_batch, local_labels = local_batch.to(device), local_labels.to(device)
			start_time=time.time()

			# Model computations
			optimizer.zero_grad()
			local_batch, local_labels=Variable(local_batch), Variable(local_labels)

			# forward + shapes modification...
			local_batch=local_batch.unsqueeze_(1)
			local_batch=local_batch.type(torch.cuda.FloatTensor)
			outputs = model(local_batch)
			_, preds = torch.max(outputs, 1)
			loss = criterion(outputs, local_labels)


			# backward + optimize only if in training phase
			# zero the parameter gradients
			a = list(model.parameters())[12].clone()

			loss.backward()
			for p,n in enumerate(model.parameters()):
				print n,p
			
			optimizer.step()
			scheduler.step()
			b = list(model.parameters())[12].clone()
			print torch.equal(a.data, b.data)
			# statistics
			running_loss += loss.item() * local_batch.size(0)
			dataseize+= local_batch.size(0)
			running_corrects += torch.sum(preds == local_labels.data)
			corrects=torch.sum(preds == local_labels.data)
			total=local_batch.size(0)
			acc=float(corrects)/float(total)
			sys.stdout.write('\r%s %s %s %s %s %s %s %s' % ('Processing training batch: ', cont, '/', training_generator.__len__(),' with loss: ', loss.item(),' and acc: ',acc)),
			sys.stdout.flush()

		epoch_loss = running_loss / dataseize
		epoch_acc = running_corrects.double() / dataseize

		print('\n{} Loss: {:.4f} Acc: {:.4f}'.format("Train", epoch_loss, epoch_acc))


		        
		model.eval()   # Set model to evaluate mode

		running_loss = 0.0
		running_corrects = 0
		dataseize=0
		# Validation
		for local_batch, local_labels in validation_generator:
			# Transfer to GPU
			# forward + shapes modification...
			ini=time.time()
			local_batch=local_batch.unsqueeze_(1)
			local_batch=local_batch.type(torch.FloatTensor)
			local_batch, local_labels = local_batch.to(device), local_labels.to(device)
			outputs = model(local_batch)
			_, preds = torch.max(outputs, 1)

			loss = criterion(outputs, local_labels)


			# statistics
			running_loss += loss.item() * local_batch.size(0)
			running_corrects += torch.sum(preds == local_labels.data)
			dataseize+= local_batch.size(0)
			corrects=torch.sum(preds == local_labels.data)
			total=local_batch.size(0)
			acc=float(corrects)/float(total)
			sys.stdout.write('\r%s %s %s %s %s %s %s %s' % ('Processing val batch: ', cont, '/', validation_generator.__len__(),' with loss: ', loss.item(),' and acc: ',acc)),
			sys.stdout.flush()


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
	return model


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


d_fmaps = [16, 32, 128, 256, 512, 1024]

model_ft = MiniVGG()#MiniVGG()

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()


# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model_ft.parameters(),lr=1e-3, weight_decay=5e-5)#  L2 regularization
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
