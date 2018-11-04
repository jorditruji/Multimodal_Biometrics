POWER_SPECTRUM_FLOOR = 1e-8

from numpy import *



def hamming(n):
    """ Generate a hamming window of n points as a numpy array.  """
    return 0.54 - 0.46 * cos(2 * pi / n * (arange(n) + 0.5))

class MFCCExtractor(object):

    def __init__(self, fs, win_length_ms=35, win_shift_ms=10, FFT_SIZE=256, n_bands=256, verbose = False):
        self.fs = fs
        self.FFT_SIZE = FFT_SIZE
        self.FRAME_LEN = int(float(win_length_ms) / 1000 * fs)
        self.FRAME_SHIFT = int(float(win_shift_ms) / 1000 * fs)
        self.window = hamming(self.FRAME_LEN)
        self.n_bands = n_bands
        self.M, self.CF = self._mel_filterbank()

    def pre_emphasize(self, x, coef=0.95):
        '''x_emphazied[n]=x[n]- coef*x[n-1]'''
        if coef <= 0:
            return x
        x0 = np.reshape(x[0], (1,))
        diff = x[1:] - coef * x[:-1]
        concat = np.concatenate((x0, diff), axis=0)
        return concat

    def extract(self, signal):
        """
        Extract MFCC coefficients of the sound x in numpy array format.
        """
        if signal.ndim > 1:
            print "INFO: Input signal has more than 1 channel; the channels will be averaged."
            signal = mean(signal, axis=1)
        frames = (len(signal) - self.FRAME_LEN) / self.FRAME_SHIFT + 1
        feature = []
        for f in xrange(frames):
            # Windowing
            frame = signal[f * self.FRAME_SHIFT : f * self.FRAME_SHIFT +
                           self.FRAME_LEN] * self.window
            # Pre-emphasis
            frame[1:] -= frame[:-1] * 0.95
            # Power spectrum
            X = abs(fft.fft(frame, self.FFT_SIZE)[:self.FFT_SIZE / 2 + 1]) ** 2
            # Mel filtering, logarithm, DCT
            #X_mel=dot(self.M,X)
            #X[X < POWER_SPECTRUM_FLOOR] = POWER_SPECTRUM_FLOOR  # Avoid zero

            #X_mel=log(X)
            #X = dot(self.D, log(dot(self.M, X)))
            feature.append(X)
        feature = array(feature)

        # Show the MFCC spectrum before normalization
        # Mean & variance normalization
        if feature.shape[0] > 1:
            mu = mean(feature, axis=0)
            #print "mean: ", mu
            sigma = std(feature, axis=0)
            #print "std: ", sigma
            feature = (feature - mu) / sigma
        return feature

    def _mel_filterbank(self):
        """
        Return a Mel filterbank matrix as a numpy array.
        Ref. http://www.ifp.illinois.edu/~minhdo/teaching/speaker_recognition/code/melfb.m
        """
        f0 = 700.0 / self.fs
        fn2 = int(floor(self.FFT_SIZE / 2))
        lr = log(1 + 0.5 / f0) / (self.n_bands + 1)
        CF = self.fs * f0 * (exp(arange(1, self.n_bands + 1) * lr) - 1)
        bl = self.FFT_SIZE * f0 * (exp(array([0, 1, self.n_bands, self.n_bands + 1]) * lr) - 1)
        b1 = int(floor(bl[0])) + 1
        b2 = int(ceil(bl[1]))
        b3 = int(floor(bl[2]))
        b4 = min(fn2, int(ceil(bl[3]))) - 1
        pf = log(1 + arange(b1, b4 + 1) / f0 / self.FFT_SIZE) / lr
        fp = floor(pf)
        pm = pf - fp
        M = zeros((self.n_bands, 1 + fn2))
        for c in xrange(b2 - 1, b4):
            r = int(fp[c] - 1)
            M[r, c+1] += 2 * (1 - pm[c])
        for c in xrange(b3):
            r = int(fp[c])
            M[r, c+1] += 2 * pm[c]
        return M, CF

POWER_SPECTRUM_FLOOR = 1e-6

from numpy import *



def hamming(n):
    """ Generate a hamming window of n points as a numpy array.  """
    return 0.54 - 0.46 * cos(2 * pi / n * (arange(n) + 0.5))

class Spectrum_Extractor(object):
    """
    Returns the spectrum of an audio file
    Param:
        fs= sampling freq
        win_length_ms = lenght of the hamming window
        win_shift = shift of the window
        FFT_size = FFT points 
        n_bands = bands used on mel filterbank
        mel = Convert to mel scale
        normalize = normalize each freq bin aling frames
    """
    def __init__(self, fs, win_length_ms=25, win_shift_ms=10, FFT_SIZE=512, n_bands=256,mel=False, normalize=True):
        self.fs = fs
        self.FFT_SIZE = FFT_SIZE
        self.FRAME_LEN = int(float(win_length_ms) / 1000 * fs)
        self.FRAME_SHIFT = int(float(win_shift_ms) / 1000 * fs)
        self.window = hamming(self.FRAME_LEN)
        self.n_bands = n_bands
        self.mel=mel
        self.M, self.CF = self._mel_filterbank()
        self.normalize = normalize


    def extract(self, signal):
        """
        Extract Spectrum of the sound x in numpy array format.
        """
        if signal.ndim > 1:
            print "INFO: Input signal has more than 1 channel; the channels will be averaged."
            signal = mean(signal, axis=1)
        frames = (len(signal) - self.FRAME_LEN) / self.FRAME_SHIFT + 1
        feature = []
        for f in xrange(frames):
            # Windowing
            frame = signal[f * self.FRAME_SHIFT : f * self.FRAME_SHIFT +
                           self.FRAME_LEN] * self.window
            # Pre-emphasis
            frame[1:] -= frame[:-1] * 0.95
            # Power spectrum
            X = abs(fft.fft(frame, self.FFT_SIZE)[:self.FFT_SIZE / 2 + 1]) ** 2
            X[X < POWER_SPECTRUM_FLOOR] = POWER_SPECTRUM_FLOOR  # Avoid zero
            # Mel filtering, logarithm
            if self.mel:
                X_mel=dot(self.M,X)
                #X[X < POWER_SPECTRUM_FLOOR] = POWER_SPECTRUM_FLOOR  # Avoid zero
                X_mel=log(X)
                X = dot(self.D, log(dot(self.M, X)))
            feature.append(X)
        feature = 10*log(transpose(array(feature)))
        # Mean & variance normalization

        #return feature
        if feature.shape[1] > 1 and self.normalize:
            print feature.shape
            mu = mean(feature, axis=1).squeeze()
            sigma = std(feature, axis=1).squeeze()
            print "\n std: ", sigma
            print "\n mean", mu
            feature = (feature - mu) / sigma
            print "abs: ", amax(feature,axis=1)-amin(feature,axis=1)
            print "max: ", amax(feature,axis=1)

        return feature

    def _mel_filterbank(self):
        """
        Return a Mel filterbank matrix as a numpy array.
        Ref. http://www.ifp.illinois.edu/~minhdo/teaching/speaker_recognition/code/melfb.m
        """
        f0 = 700.0 / self.fs
        fn2 = int(floor(self.FFT_SIZE / 2))
        lr = log(1 + 0.5 / f0) / (self.n_bands + 1)
        CF = self.fs * f0 * (exp(arange(1, self.n_bands + 1) * lr) - 1)
        bl = self.FFT_SIZE * f0 * (exp(array([0, 1, self.n_bands, self.n_bands + 1]) * lr) - 1)
        b1 = int(floor(bl[0])) + 1
        b2 = int(ceil(bl[1]))
        b3 = int(floor(bl[2]))
        b4 = min(fn2, int(ceil(bl[3]))) - 1
        pf = log(1 + arange(b1, b4 + 1) / f0 / self.FFT_SIZE) / lr
        fp = floor(pf)
        pm = pf - fp
        M = zeros((self.n_bands, 1 + fn2))
        for c in xrange(b2 - 1, b4):
            r = int(fp[c] - 1)
            M[r, c+1] += 2 * (1 - pm[c])
        for c in xrange(b3):
            r = int(fp[c])
            M[r, c+1] += 2 * pm[c]
        return M, CF