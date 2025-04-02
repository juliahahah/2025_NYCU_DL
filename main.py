import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

from neural_network.layers import Layer, Dense
from neural_network.losses import Loss, MSE, BinaryCrossEntropy
from neural_network.opti import SGD, Momentum, Adagrad, Adam, RMSprop
from neural_network.utils import generate_linear_data, generate_xor_data, show_loss, show_result
from neural_network.activations import Sigmoid, ReLU, Linear, LeakyReLU, Activation
from argparse import ArgumentParser

# 可用的激活函數列表
# 修改這一行
activation_functions_list = ["sigmoid", "ReLU", "Tanh", "Relu", "Leaky Relu", "linear", "none"]

# Create directory for saving visualization images
os.makedirs("images", exist_ok=True)


class Sequential:
    """序列模型類別 - 將多個層按順序堆疊"""

    def __init__(self):
        self.layers = []  # 儲存網絡的所有層

    def add(self, layer):
        """添加一個層到模型中"""
        self.layers.append(layer)

    def forward(self, x):
        """前向傳播 - 依次通過所有層"""
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, grad):
        """反向傳播 - 計算梯度"""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def predict(self, x):
        """進行預測 - 僅執行前向傳播"""
        return self.forward(x)


class Model(Sequential):
    """模型類，繼承自Sequential，添加特定的網絡結構和訓練方法"""

    def __init__(self, input_size, output_size, activation, hidden_units, optimizer_name, learning_rate):
        super().__init__()

        optimizers = {
            "sgd": SGD,
            "momentum": Momentum,
            "adagrad": Adagrad,
            "rmsprop": RMSprop,
            "adam": Adam
        }
        optimizer_class = optimizers.get(optimizer_name.lower(), SGD)  # 預設使用SGD
        self.optimizer = optimizer_class(learning_rate=learning_rate)  # 創建優化器

        # 激活函數字典映射
        activations = {
            "sigmoid": Sigmoid,
            "relu": ReLU,
            "linear": Linear,
            "leakyrelu": LeakyReLU,
            "none": None  # 新增: 不使用非線性激活函數
        }

        # 獲取激活函數類，預設為 Sigmoid
        self.act = activations.get(activation.lower(), Sigmoid)
        
        # 動態創建網絡結構
        prev_size = input_size
        
        # 添加所有隱藏層
        for i, hidden_size in enumerate(hidden_units):
            # 添加隱藏層
            self.add(Dense(prev_size, hidden_size))
            
            # 只有當激活函數不是"none"時才添加激活函數
            if self.act is not None:
                self.add(self.act())
            
            prev_size = hidden_size
            
        # 添加輸出層
        self.add(Dense(prev_size, output_size))
        
        # 輸出層總是使用Sigmoid (對於二元分類問題)
        self.add(Sigmoid())

        self.loss = BinaryCrossEntropy()  # 使用二元交叉熵作為損失函數

    def train_step(self, x, y):
        """執行一步訓練，使用選定的優化器"""
        # 前向傳播
        y_pred = self.forward(x)

        # 計算損失
        loss_value = self.loss.forward(y_pred, y)

        # 反向傳播
        grad = self.loss.backward(y_pred, y)
        self.backward(grad)

        # 收集參數和梯度
        params = {}
        grads = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                for param_name in layer.params:
                    # 為每個參數創建唯一的鍵
                    key = f"layer{i}_{param_name}"
                    params[key] = layer.params[param_name]
                    grads[key] = layer.grads[param_name]

        # 使用優化器更新參數
        updated_params = self.optimizer.update(params, grads)

        # 更新層參數
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'params'):
                for param_name in layer.params:  # 遍歷每個參數
                    key = f"layer{i}_{param_name}"  # 參數鍵
                    if key in updated_params:
                        layer.params[param_name] = updated_params[key]  # 更新參數

        return loss_value


def main(args):
    # 根據選擇的問題載入數據
    if args.problem == "linear":
        X, Y = generate_linear_data(n=100)  # 生成線性可分的數據
        in_dim, out_dim = 2, 1  # 輸入維度為2，輸出維度為1
    else:
        args.problem = "XOR"  # 修正賦值
        X, Y = generate_xor_data()  # 生成XOR問題數據
        in_dim, out_dim = 2, 1

    # 顯示網絡結構
    print("\n---- 神經網络配置 ----")
    print(f"輸入層: {in_dim} 個神經元")
    for i, units in enumerate(args.hidden_units):
        act_func = args.activation if args.activation != "none" else "無激活函數"
        print(f"隐藏層 {i+1}: {units} 個神經元, 激活函數: {act_func}")
    print(f"輸出層: {out_dim} 個神經元, 激活函數: sigmoid")
    print(f"優化器: {args.optimizer}, 學習率: {args.lr}")
    print(f"訓練輪次: {args.epoch}")
    print("--------------------\n")

    # 創建模型
    model = Model(in_dim, out_dim, args.activation, args.hidden_units,
                  optimizer_name=args.optimizer, learning_rate=args.lr)

    losses_per_epoch = []  # 記錄每個epoch的損失
    for e in range(args.epoch):
        losses = []  # 記錄當前epoch的所有批次損失

        # 訓練階段
        for i in range(len(X)):
            # 調整輸入和標籤的形狀
            x = X[i].reshape(1, -1)  # 將特徵調整為(1, 特徵數)
            y = Y[i].reshape(1, -1)  # 將標籤調整為(1, 1)

            # 執行一步訓練並記錄損失
            loss = model.train_step(x, y)
            losses.append(loss)

        # 記錄本輪平均損失
        losses_per_epoch.append(np.mean(losses))

        # 驗證階段
        predictions = []
        for i in range(len(X)):
            pred = model.predict(X[i].reshape(1, -1))
            predictions.append(pred)

        # 計算準確率
        y_pred = np.array(predictions)
        y_pred = (y_pred > 0.5).astype(int)  # 二分類閾值為0.5
        acc = np.mean(y_pred.reshape(Y.shape) == Y)

        # 每500輪顯示一次進度
        if (e + 1) % 500 == 0:
            print(f"輪次 {e + 1:>4}/{args.epoch} 損失: {np.mean(losses):.8f}, 準確率: {acc:.2f}")

    # 顯示訓練過程中的損失曲線
    architecture_name = '-'.join(map(str, args.hidden_units))
    show_loss(losses_per_epoch, name=f"{args.problem}_{architecture_name}")

    # 生成預測結果用於可視化
    predictions = []
    for i in range(len(X)):
        pred = model.predict(X[i].reshape(1, -1))
        predictions.append(pred[0][0])  # 提取標量值

    # 顯示結果 - 移除不支持的name參數
    show_result(X, Y, np.array(predictions))

    # 將結果圖保存為特定名稱 (手動保存圖片)
    plt.savefig(f'images/results_{args.problem}_{architecture_name}.png', dpi=300, bbox_inches='tight')
    
    # 保存模型
    if args.save_model:
        model_filename = f"{args.problem}_{architecture_name}_model.pkl"
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"模型已保存至 {model_filename}")


if __name__ == "__main__":
    # 解析命令行參數
    Parser = ArgumentParser()
    Parser.add_argument("--problem", type=str, default="linear", choices=["linear", "XOR"], help="問題類型：線性或XOR")
    Parser.add_argument("--activation", type=str, default="sigmoid", choices=activation_functions_list, help="激活函數類型，使用'none'禁用非線性激活函數")
    Parser.add_argument("--epoch", type=int, default=30000, help="訓練輪次")
    Parser.add_argument("--lr", type=float, default=0.001, help="學習率")
    Parser.add_argument("--hidden_units", type=int, nargs='+', default=[32, 32], help="隱藏層神經元數量列表，例如: --hidden_units 5 15 25")
    Parser.add_argument("--scheduler", action="store_true", help="是否使用學習率調度器")
    Parser.add_argument("--save_model", action="store_true", help="是否保存訓練後的模型")
    Parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "momentum", "adagrad", "rmsprop", "adam"], help="優化器類型")
    args = Parser.parse_args()
    main(args)