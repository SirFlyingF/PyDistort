'''
    This file contains loss function as used in Wright et al, and Steinmetz et al
    See Related-Research-Papers folder for more
'''
from tensorflow.python.ops import math_ops
from keras.saving import register_keras_serializable
from scipy.linalg import norm
from scipy.signal import ShortTimeFFT as stft
from numpy import log10

class WrightLoss:
    ''' Named after author of the paper. Not sure if standard name exists'''
    def __init__(self, ratio=0.5):
        self.ratio = ratio
    
    def _DCLoss(self, y_true, y_pred):
        dc_loss = math_ops.abs(math_ops.subtract(math_ops.mean(y_true, 0), math_ops.mean(y_pred, 0)))
        dc_loss = math_ops.mean(dc_loss, axis=-1)
        dc_energy = math_ops.mean(math_ops.abs(y_true), axis=-1) + 0.00001
        dc_loss = math_ops.div(dc_loss, dc_energy)
        return dc_loss
    
    def _ESRLoss(self, y_true, y_pred):
        esr_loss = math_ops.abs(math_ops.subtract(y_pred, y_true))
        esr_loss = math_ops.mean(esr_loss, axis=-1)
        esr_energy = math_ops.mean(math_ops.abs(y_true), axis=-1) + 0.00001
        esr_loss = math_ops.div(esr_loss, esr_energy)
        return esr_loss

    def __call__(self, y_true, y_pred):
        # return value shape should be = (#examples, )
        # that is, one value for each train example
        loss =    (self.ratio)*self._ESRLoss(y_true, y_pred) \
                + (1-self.ratio)*self._DCLoss(y_true, y_pred)
        return loss

class SteinmetzLoss:
    ''' Named after author of the paper. Not sure if standard name exists'''
    def __init__(self):
        self.y_true_stft = None

    def _spectral_conv(self, y_true, y_pred):
        numerator = norm(stft.stft(y_true) - stft.stft(y_pred), ord='fro', axis=-1, keepdims=True)
        denominator = norm(stft.stft(y_true), ord='fro', axis=-1, keepdims=True)
        return numerator/denominator

    def _spectral_log_mag(self, y_true, y_pred):
        numerator = norm(log10(stft.stft(y_true) - stft.stft(y_pred)), ord=1, axis=-1, keepdims=True)
        return numerator / y_pred.shape[0]
                                            
    def _mae(self, y_true, y_pred):
        return math_ops.mean(math_ops.square(y_true - y_pred))
    
    def __call__(self, y_true, y_pred):
        # If batch size is know, we can pe calculate stft.
        # no need to calculate multiple times
        #if not self.y_true_stft:
        #   self.y_true_stft = stft.stft(y_true)

        return self._spectral_conv(y_true, y_pred) + self._spectral_log_mag(y_true, y_pred) + self._mae(y_true, y_pred)