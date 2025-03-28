import numpy as np

"""
SGD – minibatch
Momentum
Adagrad
Adam
RMSprop
"""

class Optimizer:
    """優化器的基礎類別"""
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate  # 學習率
    def update(self, params, grads):
        """根據梯度更新參數"""
        raise NotImplementedError("每個優化器都必須實現update方法")


class SGD(Optimizer):
    """隨機梯度下降優化器（支持小批次）
    最基本的優化算法，直接按照梯度方向以學習率為步長更新參數。
    """
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)
    
    def update(self, params, grads):
        """使用SGD更新參數"""
        for key in params:
            params[key] -= self.lr * grads[key]  # 參數更新：減去學習率乘以梯度
        return params


class Momentum(Optimizer):
    """動量優化器
    通過累積過去梯度的動量，可以加速收斂並減少震盪。
    """
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum  # 動量係數，控制累積的歷史梯度的影響
        self.v = {}  # 速度（累積的梯度）
    
    def update(self, params, grads):
        """使用動量方法更新參數"""
        if not self.v:
            for key in params:
                self.v[key] = np.zeros_like(params[key])  # 初始化速度
                
        for key in params:
            # 更新速度：保留舊速度的動量部分，加上當前梯度的影響
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            # 使用速度更新參數
            params[key] += self.v[key]
        
        return params


class Adagrad(Optimizer):
    """Adagrad優化器
    自適應學習率方法，對頻繁更新的參數降低學習率，對不常更新的參數提高學習率。
    """
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon  # 小常數，避免除以零
        self.h = {}  # 累積的平方梯度
    
    def update(self, params, grads):
        """使用Adagrad方法更新參數"""
        if not self.h:
            for key in params:
                self.h[key] = np.zeros_like(params[key])  # 初始化累積平方梯度
                
        for key in params:
            # 累積梯度的平方
            self.h[key] += np.square(grads[key])
            # 更新參數：學習率除以累積平方梯度的平方根
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + self.epsilon)
        
        return params


class RMSprop(Optimizer):
    """RMSprop優化器
    解決Adagrad學習率過度衰減的問題，使用移動平均來代替直接累積。
    """
    def __init__(self, learning_rate=0.01, decay_rate=0.99, epsilon=1e-8):
        super().__init__(learning_rate)
        self.decay_rate = decay_rate  # 衰減率，控制歷史梯度的影響
        self.epsilon = epsilon  # 小常數，避免除以零
        self.cache = {}  # 平方梯度的衰減平均
    
    def update(self, params, grads):
        """使用RMSprop方法更新參數"""
        if not self.cache:
            for key in params:
                self.cache[key] = np.zeros_like(params[key])  # 初始化緩存
        
        for key in params:
            # 更新平方梯度的移動平均
            self.cache[key] = self.decay_rate * self.cache[key] + (1 - self.decay_rate) * np.square(grads[key])
            # 更新參數
            params[key] -= self.lr * grads[key] / (np.sqrt(self.cache[key]) + self.epsilon)
        
        return params


class Adam(Optimizer):
    """Adam優化器
    結合了Momentum和RMSprop的優點，使用一階和二階矩估計。
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1  # 一階矩估計的指數衰減率
        self.beta2 = beta2  # 二階矩估計的指數衰減率
        self.epsilon = epsilon  # 小常數，避免除以零
        self.m = {}  # 一階矩估計（梯度的移動平均）
        self.v = {}  # 二階矩估計（梯度平方的移動平均）
        self.t = 0   # 時間步
    
    def update(self, params, grads):
        """使用Adam方法更新參數"""
        if not self.m:
            for key in params:
                self.m[key] = np.zeros_like(params[key])  # 初始化一階矩估計
                self.v[key] = np.zeros_like(params[key])  # 初始化二階矩估計
        
        self.t += 1  # 更新時間步
        
        for key in params:
            # 更新一階矩估計（梯度的移動平均）
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            # 更新二階矩估計（梯度平方的移動平均）
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * np.square(grads[key])
            
            # 計算偏差修正後的一階矩估計
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            # 計算偏差修正後的二階矩估計
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # 更新參數
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return params
