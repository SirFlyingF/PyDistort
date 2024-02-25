'''
    This file contains class to process data in frequency domain.
    Currently not implemented
'''

from scipy.io.wavfile import read, write
from os.path import join
import numpy as np


class FreqDomain():
    def __init__(self, rate=44100, sample_sec=1, data_dir='Data'):
        # Define constants
        self.RATE = rate
        self.SAMPLE_SEC = sample_sec #Sample length in seconds
        self.DATA_DIR = data_dir
    

    def sample_cnt(self):
        # returns length of sample in 'count of samples'
        # cast to int, else float is passed to list index if SAMPLE_SEC < 1
        return int(self.SAMPLE_SEC * self.RATE)


    def _sample(self, old, example_cnt):
        # Splits the audio file into disjoint clips to be used as training examples
        pass
    

    def pre_process(self, filepath):
        # Returns .wav file from filepath as a split np.ndarray and its shape
        # Note : .wav bounced using LogicPro / Garageband adds 'LGWV' and 'cue ' chunks which causes
        #        scipy to throw a "WavFileWarning ... skipping it". Ignore it.
        pass
    

    def post_process(self, pred):
        # Concat the predicited examples into one np.ndarray that can be
        # saved as a .wav file
        pass
