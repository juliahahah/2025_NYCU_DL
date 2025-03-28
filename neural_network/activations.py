""""
Sigmoid
Tanh
Relu
Leaky Relu

"""

import numpy as np

class Activation:
    """Base activation function class."""
    
    def __init__(self):
        self.inputs = None
    
    def forward(self):
        """Forward pass."""
        raise NotImplementedError
    
    def backward(self):
        """Backward pass."""
        raise NotImplementedError


class Sigmoid(Activation):
    """Sigmoid activation function: f(x) = 1 / (1 + exp(-x))"""
    
    def forward(self, inputs):
        self.inputs = inputs
        return 1.0 / (1.0 + np.exp(-inputs))
    
    def backward(self, grad):
        outputs = self.forward(self.inputs)
        return grad * outputs * (1 - outputs)


class ReLU(Activation):
    """ReLU activation function: f(x) = max(0, x)"""
    
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)
    
    def backward(self, grad):
        return grad * (self.inputs > 0)
    
class LeakyReLU(Activation):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        self.a = x
        return np.maximum(self.alpha * x, x)
    
    def backward(self, x):
        return (x > 0).astype(int) + self.alpha * (x <= 0).astype(int)


class Linear(Activation):
    """Linear activation function: f(x) = x"""
    
    def forward(self, inputs):
        self.inputs = inputs
        return inputs
    
    def backward(self, grad):
        return grad
