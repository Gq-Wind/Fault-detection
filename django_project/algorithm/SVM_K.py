import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
import joblib
import numpy as np

# 定义分类器
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC


# 返回report字典和macroF1值还是返回精确率、召回率、macroF1
def train(train_set_path, model_path, num=None):  # 训练文件路径，模型保存绝对路径
    svm_model = SVC(kernel='rbf', C=4.0, gamma='scale', probability=True, random_state=42)
    # svm_model = SVC(kernel='linear', C=0.9, verbose=False, random_state=42)

    # 加载数据集
    data = pd.read_csv(train_set_path)  # 'preprocess_train.csv'

    # 数据预处理：缺失值填充和标准化处理
    imputer = SimpleImputer(strategy='mean')  # 使用均值填充缺失值
    data_imputed = imputer.fit_transform(data.iloc[:, 1:-1])
    scaler = StandardScaler()  # 进行标准化处理
    data_scaled = scaler.fit_transform(data_imputed)
    data_processed = pd.concat([pd.DataFrame(data_scaled), data.iloc[:, -1]], axis=1)

    # 划分训练集和测试集
    train_data, test_data = train_test_split(data_processed, test_size=0.1, random_state=42)

    # 训练集成分类器并进行交叉验证
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # 定义 5 折交叉验证

    # 进行交叉验证
    scores = []
    i = 0
    for train_idx, val_idx in kf.split(train_data.iloc[:, :-1], train_data.iloc[:, -1]):
        X_train, y_train = train_data.iloc[train_idx, :-1], train_data.iloc[train_idx, -1]
        X_val, y_val = train_data.iloc[val_idx, :-1], train_data.iloc[val_idx, -1]
        svm_model.fit(X_train, y_train)
        score = svm_model.score(X_val, y_val)
        scores.append(score)
        i = i + 1
        # print("k={},scores={}".format(i, score))
    scores = np.array(scores)
    # print("Cross Validation Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

    # 计算分类报告
    y_pred = svm_model.predict(test_data.iloc[:, :-1])
    y_test = test_data.iloc[:, -1]
    report = classification_report(y_test, y_pred, digits=2, zero_division=0, output_dict=True)
    # print(report)

    # 提取分类报告中的精确率、召回率和 F1 分数
    macro_precision = report['macro avg']['precision']
    macro_recall = report['macro avg']['recall']

    # 按中国软件杯方法 计算 MacroF1
    macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)

    # 打印 MacroF1
    # print("MacroF1:", macro_f1)

    # 保存模型
    joblib.dump(svm_model, model_path)

    return macro_precision, macro_f1

