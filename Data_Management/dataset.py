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
        self.mfcc = True
        self.preprocessing=False

    def __len__(self):
        '''Denotes the total number of samples'''
        return len(self.list_IDs)


    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        try:
            ID = self.list_IDs[index]
            y=self.labels[ID]
            ID=path_img2wav(ID)
            ID=ID.replace('\n','')
        except:
            print("errors")
        # Load data and get label
        fm, wav_data = wavfile.read(str(filter(lambda x: x in printable, ID).replace('youtubers_audios_audios', 'youtubers_videos_audios').replace('.png', '.wav')))
        if fm != 16000:
            raise ValueError('Sampling rate is expected to be 16kHz!')
        

        # Some preprocessing
        #if self.preprocessing:
        wav_data = self.abs_normalize_wave_minmax(wav_data)
        wav_data = self.pre_emphasize(wav_data)
            #MFCC extraction
        if self.mfcc:
            mfcc_matric=librosa.feature.mfcc(wav_data,fm,n_mfcc=64)
            return mfcc_matric

        return wav_data, y


    def abs_normalize_wave_minmax(self, wavdata):
        '''normalize'''
        x = wavdata.astype(np.int32)
        imax = np.max(np.abs(x))
        x_n = x / imax
        return x_n

    def pre_emphasize(self, x, coef=0.95):
        '''x_emphazied[n]=x[n]- coef*x[n-1]'''
        if coef <= 0:
            return x
        x0 = np.reshape(x[0], (1,))
        diff = x[1:] - coef * x[:-1]
        concat = np.concatenate((x0, diff), axis=0)
        return concat



