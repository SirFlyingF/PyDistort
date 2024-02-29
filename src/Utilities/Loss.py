'''
    This file contains loss function as used in Wright et al, and Steinmetz et al
    See Related-Research-Papers folder for more
'''
from keras.saving import register_keras_serializable
from tensorflow.signal import stft, hann_window
from tensorflow import norm
from tensorflow.math import log, abs, reduce_mean, divide

#@register_keras_serializable
class WrightLoss:
    ''' Named after author of the paper. Not sure if standard name exists'''
    def __init__(self, ratio=0.5):
        self.ratio = ratio
        self.eps = 0.00001

    def get_config(self):
        return {'ratio' : self.ratio}

    def from_config(self, config):
        self.ratio = config['ratio']
    
    def _DCLoss(self, y_true, y_pred):
        dc_loss = abs(reduce_mean(y_true, axis=0) - reduce_mean(y_pred, axis=0))
        dc_loss = reduce_mean(dc_loss, axis=-1)
        dc_energy = reduce_mean(abs(y_true), axis=-1) + self.eps
        dc_loss = divide(dc_loss, dc_energy)
        return dc_loss
    
    def _ESRLoss(self, y_true, y_pred):
        esr_loss = abs(y_pred - y_true)
        esr_loss = reduce_mean(esr_loss, axis=-1)
        esr_energy = reduce_mean(abs(y_true), axis=-1) + self.eps
        esr_loss = divide(esr_loss, esr_energy)
        return esr_loss

    def __call__(self, y_true, y_pred):
        loss =    (self.ratio)*self._ESRLoss(y_true, y_pred) \
                + (1-self.ratio)*self._DCLoss(y_true, y_pred)
        return loss

#@register_keras_serializable
class SteinmetzLoss:
    ''' Named after author of the paper. Not sure if standard name exists'''
    def __init__(self):
        self.eps = 0.00001

    def _stft(self, x):
        stft_ = stft(
                    signals = x, 
                    frame_length = 4096, # window len in no of timesteps
                    frame_step = 512, # hops in no of timesteps
                    fft_length=None, # frame length, if None, smallest power of 2 enclosing frame_length
                    window_fn=hann_window, # window type
                    pad_end=False,
                    name=None
                )
        # Output is of shape (examples, frames, bins=fft_length//2+1)
        stft_ = abs(stft_) + self.eps
        return stft_

    def get_config(self):
        return {}

    def from_config(self):
        pass

    def _spectral_conv(self, y_true, y_pred):
        numerator = norm(self._stft(y_true) - self._stft(y_pred), axis=-1)
        denominator = norm(self._stft(y_true), axis=-1) + self.eps
        return numerator / denominator

    def _spectral_log_mag(self, y_true, y_pred):
        numerator = norm(log(self._stft(y_true)) - log(self._stft(y_pred)), axis=-1)
        return numerator / numerator.shape[1]

    def _mae(self, y_true, y_pred):
        loss = reduce_mean(abs(y_pred-y_true), axis=-1)
        return loss
    
    def __call__(self, y_true, y_pred):
        # shapes being passed was (examples, timesteps, 1). stft thus calculated on last axis ending in 0 frames
        # Thus had to drop the final axis.
        y_true, y_pred = y_true[..., 0], y_pred[..., 0]
        return    reduce_mean(self._spectral_conv(y_true, y_pred) + self._spectral_log_mag(y_true, y_pred), axis=-1)\
                + self._mae(y_true, y_pred)