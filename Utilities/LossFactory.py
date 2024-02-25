'''
    This file contains Loss factory producing custom loss functions 
    made for Audio Processing purposes. 
'''
from tensorflow.python.ops import math_ops


class LossFactory:
    def __init__(self):
        pass

    def _ESRLoss(self, y_true, y_pred):
        # Read more about ESRLoss in this paper
        # https://www.researchgate.net/figure/The-validation-error-to-signal-ratio-ESR-as-a-function-of-the-processing-speed-using_fig4_338723685#:~:text=The%20ESR%20is%20the%20squared,amplifiers%20%5B23%2C%2027%5D%20.
        loss = math_ops.squared_difference(y_pred, y_true) 
        loss = math_ops.mean(loss, axis=-1)
        energy = math_ops.mean(math_ops.pow(y_true, 2), axis=-1) + 0.00001
        loss = math_ops.div(loss, energy)
        return loss
    
    def _DCLoss(self, y_true, y_pred):
        # Read more about DC bias on wikipedia at:
        # https://en.wikipedia.org/wiki/DC_bias
        loss = math_ops.pow(math_ops.subtract(math_ops.mean(y_true, 0), math_ops.mean(y_pred, 0)), 2)
        loss = math_ops.mean(loss, axis=-1)
        energy = math_ops.mean(math_ops.pow(y_true, 2), axis=-1) + 0.00001
        loss = math_ops.div(loss, energy)
        return loss

    def manufacture(self, ratio=1):
        # Returns a callable returning weighted sum of ESRLoss and DCLoss.
        # Takes in a ratio argument to calculate weghts applied to ESRLoss
        # Returned callable is NOT keras serializable by default
        loss =  lambda y_true, y_pred : (
                    (ratio)*self._ESRLoss(y_true, y_pred) + (1-ratio)*self._DCLoss(self, y_true, y_pred)
                )
        return loss