from load_dataset import *
from keras.utils import np_utils
from python_speech_features import mfcc
from python_speech_features import logfbank
from keras.layers.recurrent import LSTM


class dataset:
    """
        Class Dataset:
            - Parameters:
                - X_train, X_val => train and val samples
                - labels_train, labels_val => train and val labels
                - n_classes => number of classes
                - n_representations => number of exemplar representations per class
                - total => total number of exemplars
                - out_dim => dimensions of data (Theano 'th' or Tensorflow 'tf')

    """

    def __init__(self, path, n_classes, n_representations):
        self.X_train = []
        self.labels_train = []
        self.X_val = []
        self.labels_val = []
        self.labels = []
        self.root_path = path  # Ha de ser desde la carpeta on hi han totes les classes
        self.n_classes = n_classes
        self.n_representations = n_representations  # number of representations for each class, the representations will be splited in 3 (train, val, test)
        self.total = n_representations * n_classes
        self.names_class=[]

    def abs_normalize_wave_minmax(self, wavdata):
        x = wavdata.astype(np.int32)
        imax = np.max(np.abs(x))
        x_n = x / imax
        return x_n

    def pre_emphasize(self, x, coef=0.95):
        if coef <= 0:
            return x
        x0 = np.reshape(x[0], (1,))
        diff = x[1:] - coef * x[:-1]
        concat = np.concatenate((x0, diff), axis=0)
        return concat