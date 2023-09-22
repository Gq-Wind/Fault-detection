import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
import pickle


# 返回值没确定
def train(train_set_path, model_path, num=None):  # 训练文件路径，模型保存绝对路径
    # 加载数据集
    data = pd.read_csv(train_set_path)  # 训练文件路径 'preprocess_train.csv'

    # 数据预处理：缺失值填充和标准化处理
    imputer = SimpleImputer(strategy='mean')  # 使用均值填充缺失值
    data_imputed = imputer.fit_transform(data.iloc[:, 1:-1])
    scaler = StandardScaler()  # 进行标准化处理
    data_scaled = scaler.fit_transform(data_imputed)
    data_processed = pd.concat([pd.DataFrame(data_scaled), data.iloc[:, -1]], axis=1)

    # 权重调整
    class_counts = data_processed.iloc[:, -1].value_counts()
    class_weights = {cls: sum(class_counts) / count for cls, count in class_counts.items()}

    # 划分训练集和测试集
    train_data, test_data = train_test_split(data_processed, test_size=0.1, random_state=42)
    # 训练模型并进行交叉验证
    rf_model = RandomForestClassifier(n_estimators=133, random_state=42)  # 构建随机森林模型, class_weight=class_weights
    '''
    scores = cross_val_score(rf_model, train_data.iloc[:, :-1], train_data.iloc[:, -1], cv=5)  # 进行交叉验证
    print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    '''
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 定义 5 折交叉验证
    # 进行交叉验证
    scores = []
    i = 0
    for train_idx, val_idx in kf.split(train_data.iloc[:, :-1], train_data.iloc[:, -1]):
        X_train, y_train = train_data.iloc[train_idx, :-1], train_data.iloc[train_idx, -1]
        X_val, y_val = train_data.iloc[val_idx, :-1], train_data.iloc[val_idx, -1]
        rf_model.fit(X_train, y_train)
        score = rf_model.score(X_val, y_val)
        scores.append(score)
        i = i + 1
        # print("k={},scores={}".format(i, score))
    scores = np.array(scores)
    # print("Cross Validation Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

    y_pred = rf_model.predict(test_data.iloc[:, :-1])
    y_test = test_data.iloc[:, -1]
    # 计算分类报告
    report = classification_report(y_test, y_pred, digits=2, zero_division=0, output_dict=True)  # ,output_dict=True
    # print(report)

    # 提取分类报告中的精确率、召回率和 F1 分数
    macro_precision = report['macro avg']['precision']
    macro_recall = report['macro avg']['recall']

    # 按中国软件杯方法 计算 MacroF1
    macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)

    # 打印 MacroF1
    # print("MacroF1:", macro_f1)

    # 保存模型到文件
    model_file = model_path  # 模型保存路径 'rf_model_133_5k_0.9.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(rf_model, f)

    return macro_precision, macro_f1


'''
# 在测试集上评估模型性能
rf_model.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1])  # 训练模型
test_accuracy = rf_model.score(test_data.iloc[:, :-1], test_data.iloc[:, -1])  # 在测试集上计算准确率
print("Test Accuracy: %0.2f" % test_accuracy)
'''
'''
# 按照类别将测试集进行划分
test_by_class = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
for _, row in test_data.iterrows():
    label = int(row.iloc[-1])
    test_by_class[label].append(row.iloc[:-1])

# 分别计算每个类别在测试集中的准确率
for label, test_set in test_by_class.items():
    X_test = np.array(test_set)
    y_test = [label] * len(X_test)
    test_accuracy = rf_model.score(X_test, y_test)
    print("Label %d Test Accuracy: %0.2f" % (label, test_accuracy))
'''
