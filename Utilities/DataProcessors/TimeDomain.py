'''
    This file contains class to process data in time domain.
'''

from scipy.io.wavfile import read, write
from os.path import join
import numpy as np

class TimeDomain():
    def __init__(self, rate=44100, sample_sec=1, data_dir='Data'):
        # Define constants
        self.RATE = rate
        self.SAMPLE_SEC = sample_sec #Sample length in seconds
        self.DATA_DIR = data_dir
    

    def _timestep_per_sample(self):
        # returns length of sample in 'count of samples'
        # cast to int, else float is passed to list index if SAMPLE_SEC < 1
        return int(self.SAMPLE_SEC * self.RATE)


    def _sample(self, old, example_cnt):
        # Splits the audio file into disjoint clips to be used as training examples
        new = np.empty((example_cnt, self._timestep_per_sample()))
        i=0
        for s in range(0, old.shape[0], self._timestep_per_sample()):
            new[i] = old[s: s + self._timestep_per_sample()]
            i+=1
        return new
    

    def pre_process(self, filepath):
        # Returns .wav file from filepath as a split np.ndarray and its shape
        # Note : .wav bounced using LogicPro / Garageband adds 'LGWV' chunk which causes
        #        scipy to throw a "WavFileWarning ... skipping it". Ignore it.
        _, wavfile = read(join(self.DATA_DIR, filepath))
        example_cnt = wavfile.shape[0] // self._timestep_per_sample()
        
        # Convert to Mono by dropping the second chanel (or first chanel, doesn't matter),
        # Then truncate to a multiple of sample length, ie, ignore partial sample left at the end
        wavfile = np.delete(wavfile, 1, 1)
        wavfile = [x[0] for x in wavfile[0: example_cnt*self._timestep_per_sample()]]
        wavfile = np.array(wavfile)
        
        # Reshape into dimentions for LSTM
        # Input dimentions for LSTM (#.examples, #.timesteps, #.features)
        wavfile = (self._sample(wavfile, example_cnt)).reshape(example_cnt, self._timestep_per_sample(), -1)
        
        return wavfile.shape, wavfile
    

    def post_process(self, pred):
        # Concat the predicited examples into one np.ndarray that can be
        # saved as a .wav file
        wavfile = np.empty((pred.shape[0]*self._timestep_per_sample()))
        for s in range(pred.shape[0]):
            for i in range(self._timestep_per_sample()):
                # Reduce volume by factor of 1000. Output otherwise is too loud
                wavfile[(s*self._timestep_per_sample())+i] = pred[s][i][0] // 1000
        return wavfile
    
    def save(self, filepath, data):
        try:
            write(join(self.DATA_DIR, filepath), self.RATE, data)
        except:
            print('Unable to save file to disk')
        finally:
            print('Saved!')
