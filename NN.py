import numpy as np

def SplitDataSet(DataSet, TargetName='Survived', SplitRate=0.8):
    '''
    將資料集切割成訓練資料與驗證資料。

    參數：
        DataSet (pd.DataFrame): 包含特徵與目標變數的資料集。
        TargetName (str): 資料集中作為目標變數的欄位名稱，預設為 'Survived'。
        SplitRate (float): 訓練資料所占的比例，介於 0 至 1 之間，預設為 0.8。

    回傳：
        tuple: 四個值分別為
            - x_train (np.ndarray): 訓練集的特徵資料。
            - y_train (np.ndarray): 訓練集的目標變數。
            - x_val (np.ndarray): 驗證集的特徵資料。
            - y_val (np.ndarray): 驗證集的目標變數。
    '''

    # 計算資料集中樣本數量
    num_example = DataSet.shape[0]

    # 隨機打亂資料集
    arr = np.arange(num_example)  # 建立索引陣列
    np.random.shuffle(arr)        # 隨機打亂索引
    DataSet = DataSet.iloc[arr]   # 按打亂的索引重新排列資料集
    DataSet = DataSet.reset_index(drop=True)  # 重置索引以保持連續性

    # 計算分割位置
    s = int(num_example * SplitRate)  # 根據分割比例計算訓練資料集的大小

    # 分割資料集為訓練集與驗證集
    train = DataSet.iloc[:s]   # 前 s 筆資料作為訓練集
    val = DataSet.iloc[s:]     # 其餘資料作為驗證集

    # 分離特徵與目標變數
    x_train = train.loc[:, train.columns != TargetName].values  # 訓練集的特徵
    y_train = train.loc[:, TargetName].values.reshape(-1, 1)    # 訓練集的目標變數
    x_val = val.loc[:, val.columns != TargetName].values        # 驗證集的特徵
    y_val = val.loc[:, TargetName].values.reshape(-1, 1)        # 驗證集的目標變數

    # 輸出資料集形狀以便檢查
    print('==========SplitDataSet=========')
    print('x_train shape : ', x_train.shape)
    print('y_train shape : ', y_train.shape)
    print('x_val shape : ', x_val.shape)
    print('y_val shape : ', y_val.shape)

    # 回傳切割後的資料
    return x_train, y_train, x_val, y_val

def ComputeMeanStd(x):
    '''
    計算資料的平均值與標準差。

    參數：
        x (np.ndarray): 輸入的特徵資料，通常是 Numpy 陣列。

    回傳：
        tuple: 包含兩個值
            - mean (np.ndarray): 資料的平均值，按列計算。
            - std (np.ndarray): 資料的標準差，按列計算。
    '''

    mean = np.mean(x, axis=0)   # 計算平均值，axis=0 表示按列進行
    std = np.std(x, axis=0) # 計算標準差，axis=0 表示按列進行
    return mean, std    # 回傳平均值與標準差

def FeatureNormalization(x, mean, std):
    '''
    使用平均值與標準差對資料進行正規化。

    參數：
        x (np.ndarray): 輸入的特徵資料，通常是 Numpy 陣列。
        mean (np.ndarray): 特徵資料的平均值。
        std (np.ndarray): 特徵資料的標準差。

    回傳：
        np.ndarray: 正規化後的特徵資料。
    '''

    nor_x = (x - mean) / std    # 使用公式 (x - mean) / std 進行正規化
    return nor_x    # 回傳正規化後的資料

def initialize_parameters(num_input, num_hidden, num_output):
    '''
    初始化神經網路的權重與偏置參數。

    參數：
        num_input (int): 輸入層的神經元數量。
        num_hidden (int): 隱藏層的神經元數量。
        num_output (int): 輸出層的神經元數量。

    回傳：
        dict: 包含初始化的權重與偏置參數。
            - "W_input" (np.ndarray): 輸入層到隱藏層的權重矩陣。
            - "b_input" (np.ndarray): 隱藏層的偏置向量。
            - "W_output" (np.ndarray): 隱藏層到輸出層的權重矩陣。
            - "b_output" (np.ndarray): 輸出層的偏置向量。
    '''
    
    W_input = np.random.randn(num_hidden, num_input) # 初始化輸入層到隱藏層的權重矩陣，使用隨機值
    b_input = np.zeros(shape=(num_hidden, 1))   # 初始化隱藏層的偏置向量，初始值為 0
    W_output = np.random.randn(num_output, num_hidden)   # 初始化隱藏層到輸出層的權重矩陣，使用隨機值
    b_output = np.zeros(shape=(num_output, 1))  # 初始化輸出層的偏置向量，初始值為 0

    # 輸出權重與偏置的形狀，用於檢查是否正確
    print('==========initialize_parameters==========')
    print('W_input shape', W_input.shape)  # 輸入層權重形狀
    print('b_input shape', b_input.shape)  # 隱藏層偏置形狀
    print('W_output shape', W_output.shape)  # 隱藏層權重形狀
    print('b_output shape', b_output.shape)  # 輸出層偏置形狀

    # 使用 assert 確保權重與偏置的形狀正確
    assert (W_input.shape == (num_hidden, num_input))  # 確保輸入層權重形狀正確
    assert (b_input.shape == (num_hidden, 1))          # 確保隱藏層偏置形狀正確
    assert (W_output.shape == (num_output, num_hidden)) # 確保隱藏層權重形狀正確
    assert (b_output.shape == (num_output, 1))         # 確保輸出層偏置形狀正確

    # 將所有參數存入字典，便於後續使用
    parameters = {"W_input": W_input,
                  "b_input": b_input,
                  "W_output": W_output,
                  "b_output": b_output}

    # 回傳參數字典
    return parameters

def activate(z, fun):
    '''
    根據選定的激活函數對輸入進行計算。

    參數：
        z (np.ndarray): 輸入資料，可以是數值、向量或矩陣。
        fun (str): 選擇的激活函數類型，支援以下選項：
            - 'relu': 修正線性單元 (Rectified Linear Unit)。
            - 'tanh': 雙曲正切函數 (Tanh)。
            - 'leakyRelu': 帶有斜率的修正線性單元。

    回傳：
        np.ndarray: 經過激活函數處理後的輸出。
    '''

    if(fun == 'relu'):  # 判斷是否選擇 'relu' 激活函數
        s = np.where(z > 0, z, 0)   # 如果 z > 0，返回 z；否則返回 0
    elif(fun == 'tanh'):   # 判斷是否選擇 'tanh' 激活函數
        s = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z)) # 計算雙曲正切函數值
    elif(fun == 'leakyRelu'):   # 判斷是否選擇 'leakyRelu' 激活函數
        s = np.where(z > 0, z, z * 0.01)    # 如果 z > 0，返回 z；否則返回 0.01 * z
    return s    # 回傳激活後的結果

def sigmoid(z):
    '''
    計算 Sigmoid 激活函數的輸出。

    參數：
        z (np.ndarray): 輸入資料，可以是數值、向量或矩陣。

    回傳：
        np.ndarray: Sigmoid 函數處理後的輸出。
    '''
    
    s = 1 / (1 + np.exp(-z))    # 計算 Sigmoid 函數值，公式為 1 / (1 + e^(-z))
    return s    # 回傳結果

def forward_propagation(X, parameters):
    '''
    執行前向傳播，即計算神經網絡中每一層的輸出。

    參數：
        X (np.ndarray): 輸入資料，形狀為 (num_input, m)，其中 m 是樣本數。
        parameters (dict): 包含權重和偏置的字典，必須包括 'W_input', 'b_input', 'W_output', 'b_output'。

    回傳：
        A_output (np.ndarray): 輸出層的激活結果，形狀為 (1, m)。
        cache (dict): 包含中間層計算結果的字典，包括 'Z_input', 'A_input', 'Z_output', 'A_output'。
    '''
    
    # 從參數中取得權重和偏置
    W_input = parameters['W_input']
    b_input = parameters['b_input']
    W_output = parameters['W_output']
    b_output = parameters['b_output']

    # 計算隱藏層的線性組合 Z_input，然後經過 Tanh 激活函數得到 A_input
    Z_input = np.dot(W_input, X) + b_input
    A_input = activate(Z_input, 'tanh')

    # 計算輸出層的線性組合 Z_output，然後經過 Sigmoid 激活函數得到 A_output
    Z_output = np.dot(W_output, A_input) + b_output
    A_output = sigmoid(Z_output)

    # 驗證輸出形狀是否正確
    assert(A_output.shape == (1, X.shape[1]))

    # 儲存中間結果以便反向傳播使用
    cache = {"Z_input": Z_input,
             "A_input": A_input,
             "Z_output": Z_output,
             "A_output": A_output}

    # 返回最終輸出和中間結果
    return A_output, cache

def backward_propagation(parameters, cache, X, Y):
    '''
    執行反向傳播，即根據損失函數的梯度更新權重和偏置。

    參數：
        parameters (dict): 包含權重和偏置的字典，必須包括 'W_input', 'b_input', 'W_output', 'b_output'。
        cache (dict): 前向傳播中儲存的中間結果，包括 'A_input', 'A_output' 等。
        X (np.ndarray): 輸入資料，形狀為 (num_input, m)。
        Y (np.ndarray): 真實標籤，形狀為 (1, m)。

    回傳：
        grads (dict): 包含每層梯度的字典，必須包括 'dW_input', 'db_input', 'dW_output', 'db_output'。
    '''
    
    # 取得樣本數 m
    m = X.shape[1]

    # 從參數中提取權重和偏置
    W_input = parameters['W_input']
    W_output = parameters['W_output']
    A_input = cache['A_input']
    A_output = cache['A_output']

    # 計算輸出層的梯度
    dZ_output = (A_output - Y)  # 梯度是輸出誤差
    dW_output = (1 / m) * np.dot(dZ_output, A_input.T)  # 更新權重
    db_output = (1 / m) * np.sum(dZ_output, axis=1, keepdims=True)  # 更新偏置

    # 計算隱藏層的梯度
    dZ_input = np.multiply(np.dot(dW_output.T, dZ_output), 1 - np.power(A_input, 2))  # Tanh 的導數
    dW_input = (1 / m) * np.dot(dZ_input, X.T)  # 更新權重
    db_input = (1 / m) * np.sum(dZ_input, axis=1, keepdims=True)  # 更新偏置

    # 儲存所有的梯度
    grads = {'dW_input': dW_input,
             'db_input': db_input,
             'dW_output': dW_output,
             'db_output': db_output}

    # 返回梯度字典
    return grads

def update_parameters(parameters, grads, learning_rate):
    '''
    根據反向傳播的梯度更新神經網絡的參數。

    參數：
        parameters (dict): 包含所有權重和偏置的字典，包括 'W_input', 'b_input', 'W_output', 'b_output'。
        grads (dict): 包含每層的梯度，必須包括 'dW_input', 'db_input', 'dW_output', 'db_output'。
        learning_rate (float): 學習率，用於控制每次參數更新的步長。

    回傳：
        parameters (dict): 更新後的參數字典。
    '''

    # 取得當前的權重和偏置
    W_input = parameters['W_input']
    b_input = parameters['b_input']
    W_output = parameters['W_output']
    b_output = parameters['b_output']

    # 取得對應層的梯度
    dW_input = grads['dW_input']
    db_input = grads['db_input']
    dW_output = grads['dW_output']
    db_output = grads['db_output']

    # 根據梯度更新權重和偏置
    W_input = W_input - learning_rate * dW_input
    b_input = b_input - learning_rate * db_input
    W_output = W_output - learning_rate * dW_output
    b_output = b_output - learning_rate * db_output

    # 更新參數字典
    parameters = {"W_input": W_input,
                  "b_input": b_input,
                  "W_output": W_output,
                  "b_output": b_output}

    # 返回更新後的參數
    return parameters

def compute_loss(Y, A_output, parameters):
    '''
    計算損失函數，衡量預測結果與真實標籤之間的誤差。

    參數：
        Y (np.ndarray): 真實標籤，形狀為 (1, m)，其中 m 是樣本數。
        A_output (np.ndarray): 預測結果，形狀為 (1, m)，是神經網絡的輸出。
        parameters (dict): 神經網絡的參數字典，雖然傳入但未在此函數中使用。

    回傳：
        loss (float): 計算出的損失值。
    '''
    
    # 計算樣本數
    N = A_output.shape[1]

    # 計算損失值 (均方誤差 MSE)
    loss = np.square(A_output - Y).mean()  # MSE
    
    # 如果需要，還可以計算均方根誤差 (RMSE)
    # loss = np.sqrt(((A_output - Y) ** 2).mean())  # RMSE

    # 確保損失是單一數值（浮點數）
    loss = float(np.squeeze(loss))
    assert(isinstance(loss, float))

    # 返回損失值
    return loss

def predict(parameters, X):
    '''
    根據訓練好的參數對輸入數據進行預測。

    參數：
        parameters (dict): 包含神經網絡的所有權重和偏置。
        X (np.ndarray): 輸入數據，形狀為 (n_features, n_samples)。

    回傳：
        predictions (np.ndarray): 預測結果，形狀為 (1, n_samples)，每個樣本的預測結果。
    '''
    
    N = X.shape[1]  # 取得樣本數
    predictions = np.zeros((1, N))  # 初始化預測結果為全零的數組

    # 執行前向傳播，取得網絡輸出 A_output 和緩存 cache
    A_output, cache = forward_propagation(X, parameters)

    # 根據預測結果進行二分類 (0 或 1)，設定閾值為 0.5
    for i in range(A_output.shape[1]):
        predictions[0, i] = np.where(A_output[0, i] >= 0.5, 1, 0)

    assert(predictions.shape == (1, N))  # 確保預測結果形狀正確

    return predictions  # 返回預測結果

def calculation_metrics(y_true, y_pred, Methods='acc'):
    '''
    計算模型的評估指標，如準確率、均方誤差等。

    參數：
        y_true (np.ndarray): 真實標籤，形狀為 (1, n_samples)。
        y_pred (np.ndarray): 預測標籤，形狀為 (1, n_samples)。
        Methods (str): 要計算的指標名稱，可以選擇 'acc', 'mse', 'rmse', 或 'mae'。

    回傳：
        指標值 (float): 計算出的指標結果。
    '''

    if(Methods == 'acc'):
        return np.mean(y_true == y_pred)    # 計算準確率
    elif(Methods == 'mse'):
        return np.mean(np.square(y_true - y_pred))  # 計算均方誤差 (Mean Squared Error)
    elif(Methods == 'rmse'):    # 計算均方根誤差 (Root Mean Squared Error)
        return np.sqrt(np.mean(np.square(y_true - y_pred)))  # 修正 RMSE 計算公式
    elif(Methods == 'mae'):
        return np.mean(np.abs(y_true - y_pred)) # 計算平均絕對誤差 (Mean Absolute Error)

def TrainModel(x_train, y_train, x_val, y_val, num_hidden, num_iterations, learning_rate):
    '''
    訓練模型，並回傳模型。

    參數：
        x_train (np.ndarray): 訓練數據特徵，形狀為 (n_features, n_train_samples)。
        y_train (np.ndarray): 訓練數據標籤，形狀為 (1, n_train_samples)。
        x_val (np.ndarray): 驗證數據特徵，形狀為 (n_features, n_val_samples)。
        y_val (np.ndarray): 驗證數據標籤，形狀為 (1, n_val_samples)。
        num_hidden (int): 隱藏層的單元數。
        num_iterations (int): 訓練迭代次數。
        learning_rate (float): 學習率。

    回傳：
        model (dict): 訓練後的模型，包括損失、準確度、參數等信息。
    '''
    
    # 儲存訓練過程中的損失和準確度
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    # 獲取輸入數據的維度
    num_input = x_train.shape[0]
    num_output = y_train.shape[0]

    # 初始化模型參數
    parameters = initialize_parameters(num_input, num_hidden, num_output)

    # 開始訓練，迭代 num_iterations 次
    for i in range(1, num_iterations + 1):
        # 訓練集的前向傳播計算
        A_output, cache = forward_propagation(x_train, parameters)
        
        # 計算訓練損失
        train_loss_Temp = compute_loss(y_train, A_output, parameters)
        
        # 反向傳播計算梯度
        grads = backward_propagation(parameters, cache, x_train, y_train)
        
        # 更新參數
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # 訓練集準確度
        Y_prediction_train = predict(parameters, x_train)
        train_acc_Temp = calculation_metrics(y_train, Y_prediction_train, Methods='acc')
        
        # 驗證集的前向傳播計算
        val_A_output, val_cache = forward_propagation(x_val, parameters)

        # 計算驗證損失
        val_loss_Temp = compute_loss(y_val, val_A_output, parameters)
        
        # 驗證集準確度
        Y_prediction_val = predict(parameters, x_val)
        val_acc_Temp = calculation_metrics(y_val, Y_prediction_val, Methods='acc')
        
        # 儲存當前訓練和驗證的損失與準確度
        train_loss.append(train_loss_Temp)
        val_loss.append(val_loss_Temp)
        train_acc.append(train_acc_Temp)
        val_acc.append(val_acc_Temp)

        # 每 100 次打印一次訓練狀態
        if i % 100 == 0:
            print('epoch:%s, Train Loss:%.2f, Acc:%.2f, Val Loss:%.2f, Acc:%.2f' %
                  (i, train_loss_Temp, train_acc_Temp, val_loss_Temp, val_acc_Temp))

    # 儲存訓練結果和參數到模型字典
    model = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "parameters": parameters,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations
    }

    return model

def TestModel(model, DataSet):
    """
    測試已訓練的模型在給定數據集上的表現，並返回預測結果。

    參數：
        model : dict
            包含已訓練模型參數的字典。通常該字典應至少包含鍵 'parameters'，
            其中存儲了進行預測所需的權重或配置。

        DataSet : array-like
            用於進行預測的輸入數據。可以是 numpy 數組、pandas DataFrame 或
            其他 `predict` 函數支持的數據結構。

    返回值：
        array-like
            模型對輸入數據集生成的預測結果。具體格式取決於 `predict` 函數的實現。
    """

    return predict(model['parameters'], DataSet)