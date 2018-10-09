import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def read_pkl(path):
	#Reads paths to img's pf the dataset
	data=pickle.load(open(path,'rb'))
	#Find the label of each observation
	labels=[sample.split('/')[5] for sample in data]
	return data,labels

def split_dataset(data,labels, train_per=0.5, val_per=0.5):
	#Splits dataset in 2 or 3 partitions (if train%+val%<1 the rest goes to test)
	if train_per+val_per<1:
		x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=1-train_per)
		x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=(val_per/(1-train_per)))
		return x_train, y_train, x_val, y_val, x_test, y_test
	else:
		x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=1-train_per)
		return x_train, y_train,x_test, y_test

def clean_path_img(path):
	'''Borrowed from https://github.com/franroldans/tfm-franroldan-wav2pix/blob/master/onehot2image_dataset.py#L57'''
	format_path = path.replace(" ", "").replace("'", "").replace('"', '').replace('(', '').replace(')', '').\
	replace('#', '').replace('&', '').replace(';', '').replace('!', '').replace(',', '').replace('$', '').replace('?', '')
	return format_path


def path_img2wav(path):
	'''Borrowed from https://github.com/franroldans/tfm-franroldan-wav2pix/blob/master/onehot2image_dataset.py#L57'''
	return path.replace("video", "audio").replace("cropped_frames", "frames").replace('.jpg', '.wav').replace('.png', '.wav')\
	.replace('cropped_face_frame', path.split('/')[7].replace('_cropped_frames', '') + '_preprocessed_frame')





data,labels=read_pkl('faces2.pkl')
x_train, y_train, x_val, y_val=split_dataset(data,labels)

print("{} train samples \n {} val samples \n {} test samples".format(len(x_train), len(x_val), 0))
y_train=np.array(y_train)
y_val=np.array(y_val)
classes=[]
count=[]

for classe in np.unique(y_train):
	print(y_val[y_val==classe])
	classes.append(classe)
	count.append(2*len(y_val[np.where(y_val==classe)]))
	print("{} samples for class {}".format(len(y_val[np.where(y_val==classe)]),classe))

ind = np.arange(len(classes)) 
plt.bar(ind, count)
plt.show()