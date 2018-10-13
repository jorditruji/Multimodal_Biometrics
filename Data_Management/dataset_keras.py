from __future__ import division

import scipy.io.wavfile as wavfile
import string
import random
import unicodedata
from data_utils import path_img2wav
import numpy as np
import librosa
from python_speech_features import mfcc
import random


printable=set(string.printable)



class Dataset():
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
        self.forbidden=list(np.load('/work/jmorera/Multimodal_Biometrics/Data_Management/forbidden.npy'))
        print "Corrupted files to avoid: {}".format(str(len(self.forbidden)))

    def __len__(self):
        '''Denotes the total number of samples'''
        return len(self.list_IDs)




    def __getitem__(self, indexs=np.arrange(self.__len__())):
        'Generates one sample of data'
        indexs=random.shuffle(indexs)
        for index in indexs:
            # Select sample
            ID = self.list_IDs[index]
            y=self.labels[ID]
            #Problems with empty wav files... if we find a forbidden we will get another random sample
            correct_sample=False
            while correct_sample ==False:
                ID=path_img2wav(ID)

                # Load data and get label
                fm, wav_data = wavfile.read(ID)
                if fm != 16000:
                    raise ValueError('Sampling rate is expected to be 16kHz!')
                

                if np.max(np.abs(wav_data))>0:
                    correct_sample=True
                else:
                    if index==self.__len__()-1:
                        index=0
                    index=index+1
                    ID = self.list_IDs[index]

            
            
            # Some preprocessing
            #if self.preprocessing:
            #wav_data = self.abs_normalize_wave_minmax(wav_data,ID)
            wav_data = self.pre_emphasize(wav_data)
            #MFCC extraction
            if self.mfcc:
                
                mfcc_matric=mfcc(wav_data,samplerate=fm,numcep=32)
                mfcc_matric=(mfcc_matric - np.mean(mfcc_matric)) / np.std(mfcc_matric)
                print np.max(mfcc_matric), np.mean(mfcc_matric), np.min(mfcc_matric)
                return mfcc_matric,y


            yield wav_data, y


    def abs_normalize_wave_minmax(self, wavdata,name):
        '''normalize'''
        x = wavdata.astype(np.float)
        imax = np.max(np.abs(x))
        if imax==0:
            self.forbidden.append(name)
            #Update forbiddens:
            np.save('/work/jmorera/Multimodal_Biometrics/Data_Management/forbidden.npy', self.forbidden, allow_pickle=True, fix_imports=True)
        try:
            x_n = x / imax
            return x_n
        except:
            return

    def pre_emphasize(self, x, coef=0.95):
        '''x_emphazied[n]=x[n]- coef*x[n-1]'''
        if coef <= 0:
            return x
        x0 = np.reshape(x[0], (1,))
        diff = x[1:] - coef * x[:-1]
        concat = np.concatenate((x0, diff), axis=0)
        return concat


