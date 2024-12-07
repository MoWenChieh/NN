import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import NN

# 建立放模型與學習曲線的資料夾
folder = "result"
if os.path.exists(folder):  # 判斷資料夾是否存在
    shutil.rmtree(folder)   # 刪除資料夾
os.makedirs(folder, exist_ok=True)  # 建立資料夾

def main():
    """
        這是訓練模型的流程主程式
    """
    
    # 載入資料集
    DataSet = pd.read_csv('train_dataset.csv')
    DataSet = DataSet.drop(['PassengerId'], axis=1)  # 去除不需要的欄位

    # 切割資料集
    x_train, y_train, x_val, y_val = NN.SplitDataSet(DataSet, TargetName='Survived', SplitRate=0.8)

    # 計算特徵的平均值與標準差，並進行正規化
    mean, std = NN.ComputeMeanStd(x_train)
    x_train = NN.FeatureNormalization(x_train, mean, std)
    x_val = NN.FeatureNormalization(x_val, mean, std)

    # 儲存正規化的參數（可以選擇保存 mean, std 以便將來使用）
    pickle.dump((mean, std), open(os.path.join(folder, 'normalization_params.pkl'), 'wb'))

    # Reshape
    x_train = x_train.T
    y_train = y_train.T
    x_val = x_val.T
    y_val = y_val.T

    # 訓練模型
    num_hidden = 30  # 隱藏層神經元數量
    num_iterations = 2000  # 訓練迭代次數
    learning_rate = 0.01  # 學習率

    model = NN.TrainModel(x_train, y_train, x_val, y_val, num_hidden,
                       num_iterations=num_iterations, learning_rate=learning_rate)
    
    # 保存模型至 pickle 文件
    pickle.dump(model, open(os.path.join(folder, 'model.pkl'), 'wb'))

    # 畫出學習曲線（訓練損失）
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_iterations + 1), model['train_loss'], label='Train Loss')
    plt.plot(range(1, num_iterations + 1), model['val_loss'], label='Validation Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Learning Curve (Loss)')
    plt.legend()

    # 畫出學習曲線（訓練準確度）
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_iterations + 1), model['train_acc'], label='Train Accuracy')
    plt.plot(range(1, num_iterations + 1), model['val_acc'], label='Validation Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve (Accuracy)')
    plt.legend()

    # 儲存學習曲線圖片
    plt.savefig(os.path.join(folder, "learning_curve.jpg"))

if __name__ == "__main__":
    main()