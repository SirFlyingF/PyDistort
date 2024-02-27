'''
    Based on "Efficient Neural Networks for Real-time Analog Aud Modelling - Steinmetz et al - 2021,
    implements a TCN block using the Keras framework
'''

from keras.layers import Conv1D, BatchNormalization, PReLU, Dense, Input, Add

class ConditionlessTCN(): 
    def __init__(self, 
                filters, 
                kernel_size=3, 
                padding="causal", 
                dilation=1,
                chanel_last=True): 
        self.filters = filters
        self.padding = padding
        self.dilation_rate = dilation
        self.kernel_size = kernel_size
        self.chanel_last = chanel_last
        '''
        if chanel_last:
            input_shape = (batch_shape, steps, channels)
            out_shape = (batch_shape, new_steps, filters)
        else:
            input_shape = (batch_shape, channels, steps)
            out_shape = (batch_shape, filters, new_steps)
        '''

    def __call__(self, prev_layer):
        tcn = prev_layer
        tcn = Conv1D(
                kernel_size=self.kernel_size,
                filters=self.filters,
                padding=self.padding,
                dilation_rate=self.dilation_rate
            )(tcn)
        tcn = BatchNormalization()(tcn)
        tcn = PReLU()(tcn)
        
        skip = Conv1D(
                filters=self.filters, 
                kernel_size=1
            )(prev_layer)
        
        out = Add(tcn, skip)
        return out
