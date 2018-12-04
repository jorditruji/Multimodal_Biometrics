from __future__ import division
import torch
from torch.utils import data
import scipy.io.wavfile as wavfile
import string
import random
import unicodedata
from data_utils import path_img2wav
import numpy as np
import librosa
from python_speech_features import mfcc
from Audio.hand_crafted_feat import MFCCExtractor, Spectrum_Extractor
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import time
printable=set(string.printable)



class Dataset(data.Dataset):
    """
    Class Dataset:
    - Parameters:
        list_IDs: Vector of image paths
        labels: Dict containing the label for each image path
    """
    def __init__(self, list_IDs, labels):
        self.labels = labels
        self.list_IDs = list_IDs
        self.mfcc = False
        self.preprocessing=True
        self.forbidden=list(np.load('/imatge/jmorera/Multimodal_Biometrics/Data_Management/forbidden.npy'))
        print "Corrupted files to avoid: {}".format(str(len(self.forbidden)))

    def __len__(self):
        '''Denotes the total number of samples'''
        return len(self.list_IDs)




    def __getitem__(self, index):
        'Generates one sample of data'
        start_time = time.time()
        # Select sample
        ID = self.list_IDs[index]

        #Problems with empty wav files... if we find a forbidden we will get another random sample
        
