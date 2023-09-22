import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import tensorflow.keras as keras
import json
def run(test_set_path, model_path, result_path):
    # 导入模型
    model = keras.models.load_model(model_path)

    # 导入数据处理空缺值
    data = pd.read_csv(test_set_path)
    # 数据预处理：缺失值填充和标准化处理
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(data.iloc[:, 1:])
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_imputed)

    # 将数据转换为张量并进行CNN的输入格式处理
    x_test = data_scaled.reshape(data_scaled.shape[0], data_scaled.shape[1], 1)

    # 运行模型进行预测
    output = model.predict(x_test)
    result = output.argmax(axis=1)  # 获取预测结果的索引

    # 统计每个标签的数量，生成字典
    cnt_dict = {}
    for label in result:
        if label not in cnt_dict:
            cnt_dict[int(label)] = 1
        else:
            cnt_dict[int(label)] += 1

    i = 0
    result_dict = {}
    for sample_id in data['sample_id']:
        result_dict[sample_id] = int(result[i])
        i = i + 1

    # 存储结果文件到指定位置
    with open(result_path, "w") as f:
        json.dump(result_dict, f)

    res = [result_dict, cnt_dict]
    return res