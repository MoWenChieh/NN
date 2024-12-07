import os
import pandas as pd
import numpy as np
import pickle
import NN

# 讀取模型
# 讀取訓練資料的平均值與標準差
# 放預測結果的資料夾
folder = "result"

def main():
    """
        這是測試模型的流程主程式
    """

    # 載入資料集
    DataSet = pd.read_csv('test_dataset.csv')
    DataSet = DataSet.drop(['PassengerId'], axis=1)  # 去除不需要的欄位

    # 讀取訓練資料的平均值與標準差，並進行正規化
    with open(os.path.join(folder, 'normalization_params.pkl'), 'rb') as file:
        mean, std = pickle.load(file)
    DataSet = NN.FeatureNormalization(DataSet, mean, std)

    # Reshape
    DataSet = DataSet.T

    # 載入模型
    with open(os.path.join(folder, 'model.pkl'), 'rb') as file:
        model = pickle.load(file)

    # 預測結果
    result = NN.TestModel(model, DataSet)

    # 確保 result 是 1D 陣列
    if isinstance(result, pd.DataFrame):
        result = result.values.flatten()  # 將 DataFrame 轉為 1D 陣列
    elif isinstance(result, list):
        result = [item for sublist in result for item in sublist]  # 展平 nested list
    elif isinstance(result, np.ndarray) and result.ndim > 1:
        result = result.flatten()  # 展平為 1D 陣列

    # 將結果寫入 CSV
    result_df = pd.DataFrame({'Prediction_Survived': result})
    result_df.to_csv(os.path.join(folder, 'result.csv'), index=False)

if __name__ == "__main__":
    main()