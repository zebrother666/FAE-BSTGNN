import time

import scipy.io
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score,confusion_matrix,accuracy_score
from torch.utils.data import Dataset, DataLoader

import argparse
import datetime
import random
import os
import pandas as pd


class TimeSeriesDataset(Dataset):
    def __init__(self, features, labels, seq_length=12,normalize=True):
        self.seq_length = seq_length
        self.normalize = normalize
        if normalize:
            self.feature_mean = np.mean(features, axis=0)
            self.feature_std = np.std(features, axis=0)
            # 避免除零错误
            self.feature_std = np.where(self.feature_std == 0, 1e-7, self.feature_std)

            # 标准化特征
            self.features = (features - self.feature_mean) / self.feature_std
        else:
            self.features = features
        self.labels = labels

    def __len__(self):
        # 根据序列长度计算可以生成的样本数
        return len(self.features) - self.seq_length + 1

    def __getitem__(self, idx):
        # 生成时间序列样本和对应的标签
        x = self.features[idx:idx+self.seq_length]
        y = self.labels[idx+self.seq_length-1]  # 取序列最后一步的标签
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def inverse_normalize(self, data):
        if self.normalize:
            return data * self.feature_std + self.feature_mean

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

# 定义GRU模型
class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # GRU层
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # 全连接层
        self.fc = nn.Linear(hidden_size, 43)
        self.fc2=nn.Linear(43, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # GRU输出
        out, _ = self.gru(x, h0)
        # 获取最后一个时间步的输出
        out = out[:, -1, :]
        # 全连接层分类
        out = self.fc(out)#torch.Size([32, 28])
        out=self.fc2(out)
        out= self.sigmoid(out)
        return out

def train(model, train_loader, optimizer, criterion,device, epochs=6):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, target in train_loader:

            # # 将数据和目标移动到设备上
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')


def evaluate(model, test_loader, device,save_path="E:\\PycharmProjects\\last_second\\baselines\\baseline_14node"):
    model.eval()
    # 用于存储每个样本的准确率、召回率、精度和F1值
    test_accuracy = []
    test_precision = []
    test_recall = []
    test_f1 = []
    total_loss = 0
    all_true_labels = []
    all_pred_labels = []
    all_pred_probs = []  # 用于存储预测概率
    total_samples = 0
    total_time = 0.0
    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()
        for input, labels in test_loader:
            total_samples += input.size(0)
            input, targets = input.to(device), labels.to(device)
            outputs = model(input)
            # outputs, intermediate_outputs = model(input, return_intermediate=True)

            predicted = (outputs > 0.5).type(torch.int)  # 使用0.5作为阈值进行标签的二值化
            # 将模型输出和真实标签存储起来
            all_pred_probs.append(outputs.cpu().numpy())
            all_true_labels.append(targets.cpu().numpy())
            all_pred_labels.append(predicted.cpu().numpy())
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()

    avg_inference_time_ms = (end_time - start_time) / total_samples * 1000
    print(f"Average inference time per sample: {avg_inference_time_ms:.4f} ms")



    # 在循环外部进行concatenate
    all_true_labels = np.concatenate(all_true_labels, axis=0)
    all_pred_labels = np.concatenate(all_pred_labels, axis=0)
    all_pred_probs = np.concatenate(all_pred_probs, axis=0)

    # 计算评价指标
    accuracy = np.sum(all_true_labels == all_pred_labels) / (all_pred_labels.size)
    precision = precision_score(all_true_labels, all_pred_labels, average='micro')
    recall = recall_score(all_true_labels, all_pred_labels, average='micro')
    f1 = f1_score(all_true_labels, all_pred_labels, average='micro')

    # # 转换为DataFrame
    # true_labels_df = pd.DataFrame(all_true_labels, columns=[f"{i}" for i in range(all_true_labels[0].shape[0])])
    # pred_labels_df = pd.DataFrame(all_pred_labels, columns=[f"{i}" for i in range(all_pred_labels[0].shape[0])])
    # pred_probs_df = pd.DataFrame(all_pred_probs, columns=[f"{i}" for i in range(all_pred_probs.shape[1])])
    #
    # # 保存到CSV
    # true_labels_df.to_csv(os.path.join(save_path, "gru_14node_true_labels.csv"), index=False)
    # pred_labels_df.to_csv(os.path.join(save_path, "gru_14node_pred_labels.csv"), index=False)
    # pred_probs_df.to_csv(os.path.join(save_path, "gru_14node_pred_probs.csv"), index=False)

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'Test Loss: {total_loss / len(test_loader)}')

def main():
    parser = argparse.ArgumentParser()
    # dataset configurations
    parser.add_argument("--dataset", type=str, default="fdia_14", help="Dataset name.")
    # parser.add_argument("--file_path", type=str,
    #                     default="E:\\PycharmProjects\\last_second\\dataset\\8_4_node14_17000.mat",
    #                     help="Dataset filepath.")
    parser.add_argument("--file_path", type=str,
                        default="E:\\PycharmProjects\\last_second\\dataset\\MODEL_data.mat",
                        help="Dataset filepath.")
    parser.add_argument("--test_ratio", type=float, default=0.3, help="Test ratio.")
    #parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio.")
    parser.add_argument("--slide_win", type=int, default=12, help="Slide window size.")
    parser.add_argument("--slide_stride", type=int, default=1, help="Slide window stride.")
    # model configurations
    parser.add_argument("--batch_size", type=int, default=32
                        , help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=30
                        , help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.00012, help="Weight decay.")
    parser.add_argument("--device", type=int, default=0, help="Training cuda device, -1 for cpu.")
    parser.add_argument("--random_seed", type=int, default=42
                        , help="Random seed.")
    # config
    args = parser.parse_args()  # 使用 parser.parse_args() 解析命令行参数，将其保存在 args 对象中。

    # set random seed
    # 调用 seed_everything(args.random_seed) 来设置随机种子，以确保实验的可重复性。
    set_seed(args.random_seed)
    # 创建一个 PyTorch 设备对象 device，用于指定模型在 CPU 或 GPU 上运行。
    device = torch.device("cpu") if args.device < 0 else torch.device("cuda:{}".format(args.device))

    # load datasets and dataloaders
    print("Loading datasets...")

    # 加载数据
    mat = scipy.io.loadmat(args.file_path)
    # features = mat['features']
    # labels = mat['labels']
    # # 数据集划分
    # dataset = TimeSeriesDataset(features, labels, seq_length=args.slide_win, normalize=True)
    #
    # test_size =int(args.test_ratio* len(dataset))
    # train_size = len(dataset) - test_size
    #
    # # print(train_size)
    # # print(test_size)
    # # input()
    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    features_train = mat['train_features']  # 2480 x 27
    # features_train = torch.tensor(features_train, dtype=torch.float32)
    # labels = torch.tensor(labels, dtype=torch.float32)
    # features_train = torch.tensor(features_train, dtype=torch.float32).unsqueeze(1)
    labels_train = mat['train_labels']  # 2480 x 28
    # labels_train = torch.tensor(labels_train, dtype=torch.float32)
    features_test = mat['test_features']  # 2480 x 27
    # features_test = torch.tensor(features_test, dtype=torch.float32)
    # features_test = torch.tensor(features_test, dtype=torch.float32).unsqueeze(1)
    labels_test = mat['test_labels']  # 2480 x 28

    train_dataset = TimeSeriesDataset(features_train, labels_train, seq_length=args.slide_win, normalize=True)
    test_dataset = TimeSeriesDataset(features_test, labels_test, seq_length=args.slide_win, normalize=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


    # build model
    model =GRUClassifier(104, 64, 2,28)

    model = model.to(device)

    criterion = nn.BCELoss()# Binary Cross-Entropy Loss for multi-label classification
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # begin training and validation
    print("Training...")
    # Training the model
    train(model, train_loader, optimizer, criterion, device)
    # testing
    print("Testing...")
    # Evaluating the model
    evaluate(model, test_loader, device)

# 总之，这段代码实现了一个端到端的深度学习模型训练流程，包括数据加载、模型构建、优化器设置、训练循环以及验证集的评估。模型在训练过程中会保存检查点，并在验证损失不再降低时触发早停。

if __name__ == '__main__':
    main()
#Average inference time per sample: 0.0287 ms



