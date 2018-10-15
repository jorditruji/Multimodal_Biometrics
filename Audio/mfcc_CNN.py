import torch
import torch.nn as nn


class MiniVGG(nn.Module):
	def __init__(self, num_classes=27):
		super(MiniVGG,self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=4, stride=4)
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=4, stride=4)
		)
		self.conv3 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=4, stride=4)
		)
		self.fc = nn.Sequential(
			#nn.AvgPool2d(kernel_size=7, stride=1, padding=0),
			nn.Linear(128,1024),
			nn.ReLU(),
			nn.Dropout(),
			nn.Linear(1024,num_classes),
			torch.nn.Softmax())

	def forward(self, x):
		out = self.conv1(x)
		out = self.conv2(out)
		out = self.conv3(out)
		out = out.reshape(out.size(0), -1)
		out = self.fc(out)
		return out


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=27):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3))
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3))
        self.layer5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3))
        self.layer6 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(
            #nn.AvgPool2d(kernel_size=7, stride=1, padding=0),
            nn.Linear(128,1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024,num_classes),
            torch.nn.Softmax())
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

# Convolutional neural network (4 convolutional layers)
class MiniConvNet(nn.Module):
	def __init__(self, num_classes=27):
		super(MiniConvNet, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=2),
			nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

		self.layer2 = nn.Sequential(
			nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),
			nn.Dropout(0.5),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.layer3 = nn.Sequential(
			nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.layer4 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.AvgPool2d(kernel_size=5, stride=3))
			#nn.AvgPool2d(kernel_size=3, stride=4, padding=0))
		self.fc = nn.Sequential(
			nn.Linear(120,1024),
			nn.ReLU(),
			nn.Dropout(),
			nn.Linear(1024,num_classes),
			torch.nn.Softmax())

	def forward(self, x):
		print x
		out = self.layer1(x)
		out = self.layer2(out)
		print out
		out = self.layer3(out)
		out = self.layer4(out)
		out = out.reshape(out.size(0), -1)
		print out
		out = self.fc(out)
		print out
		return out



# Convolutional neural network (4 convolutional layers)
class MiniConvNet2(nn.Module):
	def __init__(self, num_classes=27):
		super(MiniConvNet2, self).__init__()
		self.layer1 = nn.Sequential(
			nn.BatchNorm2d(1),
			nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=2),
			nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4,2)))

		self.layer2 = nn.Sequential(
			nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),
			nn.Dropout(0.5),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=(4,2)))
		self.layer3 = nn.Sequential(
			nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=(4,2)))
		self.layer4 = nn.Sequential(
			nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=2),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=(4,2)))

		self.fc = nn.Sequential(
			nn.Linear(96,124),
			nn.ReLU(),
			nn.Dropout(),
			nn.Linear(124,num_classes),
			torch.nn.Sigmoid())

	def forward(self, x):
		print torch.max(x)
		
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = out.reshape(out.size(0), -1)
		out = self.fc(out)
		return out
