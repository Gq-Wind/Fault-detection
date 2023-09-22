import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import save_model


def train(train_set_path, model_path, num):
    '''
    训练模型并进行K折交叉验证

    参数：
    num特征值数量

    返回值：
    classification_report：分类报告
    macro_F1: 宏平均F1分数

    '''
    # 加载CSV数据集
    dataset = pd.read_csv(train_set_path)
    X = dataset.iloc[:, 1:-1].values  # 前107列作为训练数据
    y = dataset.iloc[:, -1:].values.ravel()  # 最后1列为目标变量

    # 对数据集进行缺失值填充，用平均值填充
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # 将类别变量转换为数字
    labelencoder = LabelEncoder()
    y = labelencoder.fit_transform(y)

    # 进行标准化处理
    sc = StandardScaler()
    X = sc.fit_transform(X)

    # 将数据集按照8:2的比例划分成训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    # 设置K折交叉验证参数
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # 构建卷积神经网络模型
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(num, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=6, activation='softmax'))

    # 编译模型
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 进行K折交叉验证
    scores = []
    for train_idx, val_idx in kfold.split(X_train):
        # 取出训练集和验证集，并进行特征缩放
        X_train_fold, y_train_fold = X_train[train_idx], y_train[train_idx]
        X_val_fold, y_val_fold = X_train[val_idx], y_train[val_idx]
        X_train_fold = sc.transform(X_train_fold)
        X_val_fold = sc.transform(X_val_fold)

        # 训练模型
        model.fit(X_train_fold, y_train_fold, epochs=90, batch_size=128, verbose=0)

        # 评估模型
        score, acc = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        scores.append(acc)

    # 输出K折交叉验证的平均精度
    # print('CV accuracy:', np.mean(scores))

    # 在测试集上测试模型性能
    X_test = sc.transform(X_test)
    test_loss, test_acc = model.evaluate(X_test, y_test)
    # print('Test accuracy:', test_acc)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)

    # 根据测试数据的真实标签和模型预测生成混淆矩阵和分类报告
    conf_mat = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # 计算每个类别的预测准确率和召回率
    class_acc = []
    class_rec = []
    for i in range(len(conf_mat)):
        acc = conf_mat[i,i]/sum(conf_mat[i,:])
        class_acc.append(acc)
        rec = conf_mat[i,i]/sum(conf_mat[:,i])
        class_rec.append(rec)
    # print('Class accuracy:', class_acc)
    # print('Class recall:', class_rec)

    avg_acc = np.mean(class_acc)
    avg_rec = np.mean(class_rec)

    # print('Average class accuracy:', avg_acc)
    # print('Average class recall:', avg_rec)

    # 计算MacroF1
    macro_F1 = (2 * avg_acc * avg_rec) / (avg_acc + avg_rec)

    # 打印MacroF1
    # print('macro_F1:', macro_F1)

    # print(class_report)
    # 保存模型
    # model.save(model_path)
    save_model(model, model_path)
    return avg_acc, macro_F1



