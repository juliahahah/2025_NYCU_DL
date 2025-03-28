import numpy as np
"""
Dense layer

"""
class Layer:
    """Base layer class."""
    
    def __init__(self):
        self.params = {}
        self.grads = {}
    
    def forward(self, inputs):
        """Forward pass."""
        pass
    
    def backward(self, grad):
        """Backward pass."""
        pass


##can use keras (32 128 512 128 32) 
class Dense(Layer):
    """Fully connected layer."""
    
    def __init__(self, input_size, output_size):
        super().__init__()
        
        # Initialize weights with small random values
        #乘以 0.01 避免梯度爆炸
        self.params = {'W': np.random.randn(input_size, output_size) * 0.01, 
              'b': np.zeros(output_size)}
     
        self.inputs = None
    
    def forward(self, inputs):
        """Forward pass: y = x · W + b"""
        self.inputs = inputs
        return np.dot(inputs, self.params['W']) + self.params['b']
    
    def backward(self, grad):
        """Backward pass."""
        # Gradient w.r.t weights: ∂L/∂W = inputs^T · grad
        self.grads['W'] = np.dot(self.inputs.T, grad)
        
        # Gradient w.r.t bias: ∂L/∂b = sum(grad, axis=0)
        self.grads['b'] = np.sum(grad, axis=0)
        
        # Gradient w.r.t inputs: ∂L/∂inputs = grad · W^T
        return np.dot(grad, self.params['W'].T)
