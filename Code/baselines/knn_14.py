import scipy.io
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import random
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
# 加载数据
# mat = scipy.io.loadmat('E:\\PycharmProjects\\last_second\\dataset\\8_4_node14_17000.mat')
mat = scipy.io.loadmat('E:\\PycharmProjects\\last_second\\dataset\\DATA_data.mat')
#mat2 = scipy.io.loadmat('E:\\PycharmProjects\\FDIA_2480\\FDIA_2480\\dataset\\processed_data\\PCA-14-data.mat')
features =mat['features']  # 2480 x 27
labels = mat['labels'] # 2480 x 28
# 数据集划分（这里直接使用了假设的features和labels变量）
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
# X_train =mat['train_features']  # 2480 x 27
#
# y_train = mat['train_labels'] # 2480 x 28
#
# X_test =mat['test_features']  # 2480 x 27
#
# y_test = mat['test_labels'] # 2480 x 28



def set_seed(seed=42):
    """设置随机数种子以确保代码的可重复性."""
    random.seed(seed)  # Python内置的random模块
    np.random.seed(seed)  # Numpy库
    torch.manual_seed(seed)  # 为CPU设置随机数种子
    torch.cuda.manual_seed(seed)  # 为所有CUDA设备设置随机数种子
    torch.cuda.manual_seed_all(seed)  # 为所有CUDA设备设置随机数种子（当使用多个GPU时）
    os.environ['PYTHONHASHSEED'] = str(seed)  # 通过环境变量设置Python哈希随机化的种子

    # 以下两行是为了确保PyTorch能尽可能地实现确定性操作
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置种子
set_seed(42)


def train_and_evaluate_knn(X_train, y_train, X_test, y_test, n_neighbors=250,save_path='E:\\PycharmProjects\\last_second\\baselines\\baseline_14node'):
    """
    训练KNN模型并在测试集上评估，计算准确度、精度、召回率和F1得分。
    :param X_train: 训练特征
    :param y_train: 训练标签
    :param X_test: 测试特征
    :param y_test: 测试标签
    :param n_neighbors: 邻居数
    :return: None
    """
    # 创建KNN模型
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # 训练模型
    knn.fit(X_train, y_train)

    # 进行预测
    predictions = knn.predict(X_test)

    # 计算评估指标
    accuracy =  np.sum(y_test == predictions) / (y_test.size)
    precision = precision_score(y_test, predictions, average='micro')
    recall = recall_score(y_test, predictions, average='micro')
    f1 = f1_score(y_test, predictions, average='micro')

    # 打印评估指标
    print(f'Overall Accuracy: {accuracy:.4f}')
    print(f'Overall Precision: {precision:.4f}')
    print(f'Overall Recall: {recall:.4f}')
    print(f'Overall F1 Score: {f1:.4f}')

    # 转换为DataFrame
    # true_labels_df = pd.DataFrame(y_test, columns=[f"{i}" for i in range(y_test[0].shape[0])])
    # pred_labels_df = pd.DataFrame(predictions, columns=[f"{i}" for i in range(predictions[0].shape[0])])
    # # pred_probs_df = pd.DataFrame(all_pred_probs, columns=[f"{i}" for i in range(all_pred_probs.shape[1])])
    #
    # # 保存到CSV
    # true_labels_df.to_csv(os.path.join(save_path, "knn_14node_true_labels.csv"), index=False)
    # pred_labels_df.to_csv(os.path.join(save_path, "knn_14node_pred_labels.csv"), index=False)
    # # pred_probs_df.to_csv(os.path.join(save_path, "pred_probs.csv"), index=False)


# 执行训练和评估
train_and_evaluate_knn(X_train, y_train, X_test, y_test)

# Overall Accuracy: 0.9533
# Overall Precision: 0.9717
# Overall Recall: 0.9090
# Overall F1 Score: 0.9393