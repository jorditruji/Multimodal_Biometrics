import torch
import torch.nn as nn

# Convolutional neural network (4 convolutional layers)
class MiniConvNet(nn.Module):
	def __init__(self, num_classes=27):
		super(MiniConvNet, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=2),
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
			nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.layer4 = nn.Sequential(
			nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.AvgPool2d(kernel_size=3, stride=4, padding=0))
		self.fc = nn.Sequential(
			nn.Linear(192,512),
			nn.ReLU(),
			nn.Dropout(),
			nn.Linear(512,num_classes),
			torch.nn.Softmax())

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = out.reshape(out.size(0), -1)
		out = self.fc(out)
		return out
