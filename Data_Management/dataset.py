import torch
from torch.utils import data


class Dataset(data.Dataset):    """
        Class Dataset:
            - Parameters:
            list_IDs: Vector of image paths
            labels: Dict containing the label for each image path

    """

    def __init__(self, list_IDs, labels):
        self.labels = labels
        self.list_IDs = list_IDs


    def __len__(self):
        '''Denotes the total number of samples'''
        return len(self.list_IDs)


    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        fm, wav_data = wavfile.read(filter(lambda x: x in printable, audio_path).replace('youtubers_audios_audios', 'youtubers_videos_audios').replace('.png', '.wav'))
        if fm != 16000:
            raise ValueError('Sampling rate is expected to be 16kHz!')
        
        # Some preprocessing
        wav_data = self.abs_normalize_wave_minmax(wav_data)
        wav_data = self.pre_emphasize(wav_data)

        y = self.labels[ID]
        return X, y


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



