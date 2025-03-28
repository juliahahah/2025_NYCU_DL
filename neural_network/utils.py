
'''
data generation
- generate_linear_data
- generate_xor_data
- show_result
- show_loss
'''
#建立二維點資料，並且標記類別
#n:點的數量
def generate_linear_data(n=100):
    import numpy as np
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

#生成XOR資料
#XOR資料是一個二維資料，標記有0和1
#0和1的分布是交叉的
def generate_xor_data():
    import numpy as np
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)
        #排除中心點
        if 0.1*i == 0.5:
            continue
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21, 1)


import matplotlib.pyplot as plt
#顯示結果
#x:輸入資料
#y:標記
#pred_y:預測結果
#將輸入資料和預測結果分別用紅色和藍色標記
#紅色表示類別0，藍色表示類別1
#結果保存在result.png
def show_result(x, y, pred_y):
    plt.cla()
    plt.clf()
    plt.subplot(1, 2, 1)  # 創建 1x2 子圖，選擇第一個
    plt.title("Ground truth")  # 設定標題

    # 繪製每個資料點
    for i in range(len(x)):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')  # 紅色圓點表示類別 0
        else:
            plt.plot(x[i][0], x[i][1], 'bo')  # 藍色圓點表示類別 1
    plt.subplot(1, 2, 2)  # 選擇第二個子圖
    plt.title("Predict result")  # 設定標題
    # 繪製每個資料點，根據預測概率上色
    for i in range(len(x)):
        if pred_y[i] < 0.5:
            plt.plot(x[i][0], x[i][1], 'ro')  # 預測為類別 0 (紅色)
        else:
            plt.plot(x[i][0], x[i][1], 'bo')  # 預測為類別 1 (藍色)
    plt.savefig("result.png")

def show_loss(losses, name = "loss"):
    plt.figure(figsize=(10, 6))
    plt.cla()
    plt.clf()
    plt.plot(losses, label = name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig("loss.png")
