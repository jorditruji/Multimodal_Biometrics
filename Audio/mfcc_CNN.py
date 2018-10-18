import torch
import torch.nn as nn
import math

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
            nn.MaxPool2d(kernel_size=2, stride=2)
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
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=6, stride=6)
        )
        self.fc = nn.Sequential(
        #nn.AvgPool2d(kernel_size=7, stride=1, padding=0),
            nn.Linear(512,2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2048,num_classes))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
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


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
		    residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out
class ReLU(nn.Hardtanh):

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 20, inplace)

    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
            + inplace_str + ')'


class DeepSpeakerModel(nn.Module):
    def __init__(self,embedding_size=512,num_classes=27,feature_dim = 64):
        super(DeepSpeakerModel, self).__init__()

        self.embedding_size = embedding_size
        self.model = myResNet(BasicBlock, [1, 1, 1, 1])
        if feature_dim == 64:
            self.model.fc = nn.Linear(512*4, self.embedding_size)
        elif feature_dim == 40:
            self.model.fc = nn.Linear(256 * 5, self.embedding_size)
        self.model.classifier = nn.Sequential(
            nn.Linear(self.embedding_size, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, num_classes)
            #nn.Softmax()
        )


    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.layer1(x)

        x = self.model.conv2(x)
        x = self.model.bn2(x)
        x = self.model.relu(x)
        x = self.model.layer2(x)

        x = self.model.conv3(x)
        x = self.model.bn3(x)
        x = self.model.relu(x)
        x = self.model.layer3(x)

        x = self.model.conv4(x)
        x = self.model.bn4(x)
        x = self.model.relu(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)
        self.features = self.l2_norm(x)
        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        alpha=10
        self.features = self.features*alpha

        #x = x.resize(int(x.size(0) / 17),17 , 512)
        #self.features =torch.mean(x,dim=1)
        #x = self.model.classifier(self.features)
        return self.model.classifier(x)

    def forward_classifier(self, x):
		features = self.forward(x)
		res = self.model.classifier(features)
		return res

class myResNet(nn.Module):

    def __init__(self, block, layers, num_classes=27):

        super(myResNet, self).__init__()

        self.relu = ReLU(inplace=True)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, layers[0])

        self.inplanes = 128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2,bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.layer2 = self._make_layer(block, 128, layers[1])
        self.inplanes = 256
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2,bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.layer3 = self._make_layer(block, 256, layers[2])
        self.inplanes = 512
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2,bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.layer4 = self._make_layer(block, 512, layers[3])

        
        self.avgpool = nn.AdaptiveAvgPool2d((1,None))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):

        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        print x.size()
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

