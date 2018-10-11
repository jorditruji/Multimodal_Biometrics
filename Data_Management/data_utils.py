import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import preprocessing


def read_pkl(path):
	#Reads paths to img's pf the dataset
	data=pickle.load(open(path,'rb'))
	#Find the label of each observation
	labels=[sample.split('/')[5] for sample in data]
	return data,labels

def split_dataset(data,labels, train_per=0.7, val_per=0.2):
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


def filter_dataset_by_classes(data, labels, min_samples=10, limit=False):
	'''Returns the dataset only with the classes that have at least min_samples'''
	classes=[]
	count=[]
	data=np.array(data)
	labels=np.array(labels)
	for classe in np.unique(labels):
		classes.append(classe)
		count.append(len(labels[np.where(labels==classe)]))
	classes=np.array(classes)
	count=np.array(count)
	interesting_classes=classes[count>=min_samples]
	print(len(interesting_classes))
	reduced_labels=[]
	reduced_data=[]
	for interest in interesting_classes:
		data_to_append=data[labels==interest]
		for i,sample in enumerate(data_to_append):
			if limit:
				if i<min_samples:
					reduced_data.append(sample)
			else:
				reduced_data.append(sample)

		data_to_append=labels[labels==interest]
		for i,sample in enumerate(data_to_append):
			if limit:
				if i<min_samples:
					reduced_labels.append(sample)
			else:
				reduced_labels.append(sample)
	return reduced_data, reduced_labels


def save_partition(data,labels,name):
	'''Saves matrixes intro numpy file'''
	data=[[x_sample, x_label] for x_sample, x_label in zip(data, name)]
	np.save(name, data, allow_pickle=True, fix_imports=True)

def load_partitions():
	#Loaddddddd
	return np.load(partition), np.load(labels)

'''
data,labels=read_pkl('faces2.pkl')
data,labels=filter_dataset_by_classes(data,labels, min_samples=300)

#Encode labels:
label_encoder = preprocessing.LabelEncoder()
labels=label_encoder.fit_transform(labels)

dict_labels={}
for label,path in zip(labels,data):
	dict_labels[path]=label

np.save('labels',dict_labels,allow_pickle=True, fix_imports=True)
#Split dataset
x_train, y_train, x_val, y_val, x_test, y_test=split_dataset(data,labels)


#Save into dict:
partition={}
labels={}
partition['train']=x_train
partition['validation']=x_val
partition['test']=x_test
print(partition)
np.save('partition',partition,allow_pickle=True, fix_imports=True)
print(partition[key] for key in partition.keys())

'''
#X_dataset=np.concatenate((x_train,y_train),axis=1)
