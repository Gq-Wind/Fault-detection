import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import joblib


def train(train_set_path, model_path, num):
    """
    训练 SVM 模型

    参数：
    dataset_path：str类型，数据集路径

    返回值：
    average_precision: 平均预测准确率
    macro_F1: 宏平均F1分数
    """

    # 加载数据集
    data = pd.read_csv(train_set_path)

    # 对缺失值进行平均值填充
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X = imputer.fit_transform(data.iloc[:, 1:-1].values)
    y = data.iloc[:, -1].values   # 标签

    # 特征标准化处理
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # 创建 SVM 分类器
    clf = SVC(kernel='linear', C=0.9, random_state=42, verbose=False)

    # 模型训练
    clf.fit(X_train, y_train)

    # 模型预测
    y_pred = clf.predict(X_test)

    # 计算分类报告
    report = classification_report(y_test, y_pred, output_dict=True)
    # print(report)

    # 提取分类报告中的精确率、召回率和 F1 分数
    macro_precision = report['macro avg']['precision']
    macro_recall = report['macro avg']['recall']

    # 按中国软件杯方法 计算 MacroF1
    macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)

    # 打印 MacroF1
    print('precision', macro_precision, "MacroF1:", macro_f1)

    # 保存模型
    joblib.dump(clf, model_path)

    # 返回平均预测准确率和MacroF1
    return macro_precision, macro_f1
