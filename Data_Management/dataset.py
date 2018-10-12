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
        #Problems with empty wav files... if we find a forbidden we will get another random sample
        forbidden=["/imatge/froldan/work/youtubers_videos_audios/SoyUnaPringada/audio/Elanoenelquecasinomequisesuicidar-SoyUnaPringada-MUKv8poAhbA_frames/Elanoenelquecasinomequisesuicidar-SoyUnaPringada-MUKv8poAhbA_preprocessed_frame_40125.wav"
                    "/imatge/froldan/work/youtubers_videos_audios/SoyUnaPringada/audio/QueridaPringada5-SoyUnaPringada-Umb5hCEBPNo_frames/QueridaPringada5-SoyUnaPringada-Umb5hCEBPNo_preprocessed_frame_22375.wav"
                    "/imatge/froldan/work/youtubers_videos_audios/SoyUnaPringada/audio/Videosqueodio3-SoyUnaPringada-gCtfChPoZjw_frames/Videosqueodio3-SoyUnaPri$gada-gCtfChPoZjw_preprocessed_frame_44825.wav"
                    "/imatge/froldan/work/youtubers_videos_audios/unicoos/audio/Operacionescombinadas01PRIMARIAunicoos-UWkvBSCBOIY_frames/Operacionescombinadas01PRIMARIAunicoos-UWkvBSCBOIY_preprocessed_frame_19803.wav"
                    "/imatge/froldan/work/youtubers_videos_audios/SoyUnaPringada/audio/Lalocadelgimnasio-SoyUnaPringada-itpb_K5tuuY_frames/Lalocadelgimnasio-SoyUnaPringada-itpb_K5tuuY_preprocessed_frame_20850.wav"
                    "/imatge/froldan/work/youtubers_videos_audios/SoyUnaPringada/audio/Lalocadelgimnasio-SoyUnaPringada-itpb_K5tuuY_frames/Lalocadelgimnasio-SoyUnaPringada-itpb_K5tuuY_preprocessed_frame_20850.wav"
                    ]

        try:
            ID = self.list_IDs[index]
            if ID==forbidden:
                index=index+3
                ID = self.list_IDs[index]

            y=self.labels[ID]
            ID=path_img2wav(ID)
            ID=ID.replace('\n','')
        except:
            print("errors")
        # Load data and get label
        fm, wav_data = wavfile.read(ID)
        if fm != 16000:
            raise ValueError('Sampling rate is expected to be 16kHz!')
        

        # Some preprocessing
        #if self.preprocessing:
        wav_data = self.abs_normalize_wave_minmax(wav_data,ID)
        wav_data = self.pre_emphasize(wav_data)
        #MFCC extraction
        if self.mfcc:
            try:
                mfcc_matric=librosa.feature.mfcc(wav_data,fm,n_mfcc=64)
                return mfcc_matric,y
            except:
                return

        return wav_data, y


    def abs_normalize_wave_minmax(self, wavdata,name):
        '''normalize'''
        x = wavdata.astype(np.int32)
        imax = np.max(np.abs(x))
        if imax==0:
            print imax,name
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



