import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import json


# 生成结果文件，返回统计结果Json字符串
def run(test_set_path, model_path, result_path):  # 测试文件路径，调用模型路径，预测结果文件路径
    # 导入模型
    model = joblib.load(model_path)  # 调用模型路径 'linear_0.9.pkl'

    # 导入数据处理空缺值
    data = pd.read_csv(test_set_path)  # 测试文件路径 'validate_1000.csv'
    # 数据预处理：缺失值填充和标准化处理
    imputer = SimpleImputer(strategy='mean')  # 使用均值填充缺失值
    data_imputed = imputer.fit_transform(data.iloc[:, 1:])
    scaler = StandardScaler()  # 进行标准化处理
    data_scaled = scaler.fit_transform(data_imputed)

    result = model.predict(data_scaled)  # 返回的预测值numpy数组
    # print(result)
    # 统计每个标签的数量，生成字典
    cnt_dict = {}
    for label in result:
        if label not in cnt_dict:
            cnt_dict[int(label)] = 1
        else:
            cnt_dict[int(label)] += 1
    # print(cnt_dict)

    i = 0
    result_dict = {}  # 生成”编号-类型“字典文件
    for sample_id in data['sample_id']:
        result_dict[sample_id] = int(result[i])
        i = i + 1
    # print(result_dict);
    # print(json.dumps(result_dict))  # 生成json字符串
    # 存储结果文件到指定位置
    with open(result_path, "w") as f:  # 结果文件路径
        json.dump(result_dict, f)
    res = [result_dict, cnt_dict]
    return res
    # return json.dumps(cnt_dict)  # 返回统计结果cnt_dict的json字符串


