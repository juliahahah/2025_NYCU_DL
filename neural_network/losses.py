import numpy as np

"""
MSE
BinaryCrossEntropy
"""
    
class Loss:
    """Base loss function class."""
    
    def __init__(self):
        pass
    
    def forward(self, y_pred, y_true):
        """Compute the loss value."""
        pass
    
    def backward(self, y_pred, y_true):
        """Compute gradient of loss w.r.t. predictions."""
        pass


class MSE(Loss):
    """Mean Squared Error loss function."""
    
    def forward(self, y_pred, y_true):
        """Compute MSE loss: (1/n) * Σ(y_pred - y_true)²"""
        return np.mean(np.power(y_pred - y_true, 2))
    
    def backward(self, y_pred, y_true):
        """Gradient of MSE: (2/n) * (y_pred - y_true)"""
        n_samples = y_true.shape[0]
        return 2 * (y_pred - y_true) / n_samples


class BinaryCrossEntropy(Loss):
    """Binary cross-entropy loss function."""
    
    def forward(self, y_pred, y_true):
        """Compute binary cross-entropy loss."""
        # Clip values to prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def backward(self, y_pred, y_true):
        """Gradient of binary cross-entropy."""
        # Clip values to prevent division by zero
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        n_samples = y_true.shape[0]
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / n_samples

