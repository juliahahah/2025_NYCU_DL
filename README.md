# NumPy Neural Network Implementation

這個專案是一個使用純 NumPy 實現的簡單神經網絡庫，可用於解決線性分類和 XOR 問題等基本機器學習任務。

## 使用方法


## Test Results
## test for linear problem
```bash
python main.py --problem linear --save_model
```

## test for XOR problem
```bash
python main.py --problem XOR --save_model
```

## 調整參數
```bash
python main.py --problem XOR --activation ReLU --epoch 30000 --lr 0.01 --hidden 32 --optimizer rmsprop --save_model
```

### 沒有act
```bash
python main.py --problem XOR --activation none --hidden_units 32 16
```