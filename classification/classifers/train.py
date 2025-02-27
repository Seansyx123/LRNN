import random
import models
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score, confusion_matrix, matthews_corrcoef, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

# 定义相关参数
device = "cuda"
batch_size = 32
epochs = 100
lr = 5e-4



stand = StandardScaler()
label_dict = {
    "clean":0,
    "buggy":1
}

name = "DNN"
model_name = f"{name}Model"

kind = " "
num_lens = 61

# 数据样本分布不均匀，进行采样
def random_data(x, y):

    # x, y = torch.tensor(x), torch.tensor(y, dtype=torch.int64)

    o_index = torch.where(y == 1)[0]
    o_x = x[o_index, :]
    o_y = y[o_index]

    index = torch.where(y == 0)[0]
    index = random.sample(index.tolist(), o_x.shape[0])

    z_x = x[index, :]
    z_y = y[index]

    x = torch.cat([o_x, z_x], dim=0)
    y = torch.cat([o_y, z_y], dim=0)

    return x, y



def read_data():
    global num_lens

    train_data = pd.read_csv("train_PDE.csv")
    test_data = pd.read_csv("test_PDE.csv")

    # 填充NAN
    train_data = train_data.fillna(0)
    test_data = test_data.fillna(0)

    # 进行标准化
    train_y = train_data.pop("class").values
    train_x = train_data.values

    test_y = test_data.pop("class").values
    test_x = test_data.values

    train_x = stand.fit_transform(train_x)
    test_x = stand.transform(test_x)

    # 对Y进行编码
    train_y = [label_dict[i] for i in train_y]
    test_y = [label_dict[i] for i in test_y]

    if kind == "pca":
        num_lens = 16
        pca = PCA(n_components=num_lens)  # 降到16维度
        train_x = pca.fit_transform(train_x)
        test_x = pca.transform(test_x)
    elif kind == "lda": # 占时没有找到强制降维的方法，这个降维只能到class_num-1，那当前只能设置为1
        # 创建并训练LDA分类器
        lda = LDA(n_components=None)  # 若不指定n_components，则默认为类别数-1

        lda.fit(train_x, train_y)

        # 使用LDA模型对测试集进行预测
        y_pred = lda.predict(test_x)

        precision = precision_score(test_y, y_pred, average="macro")
        recall = recall_score(test_y, y_pred, average="macro")
        f1 = f1_score(test_y, y_pred, average="macro")
        accuracy = accuracy_score(test_y, y_pred)
        mcc = matthews_corrcoef(test_y, y_pred)
        auc = roc_auc_score(test_y, y_pred)

        print(f"precision为：{precision * 100}%")
        print(f"recall为：{recall * 100}%")
        print(f"f1为：{f1 * 100}%")
        print(f"accuracy：{accuracy * 100}%")
        print(f"mcc：{mcc * 100}%")
        print(f"auc：{auc * 100}%")

        cm = confusion_matrix(test_y, y_pred)
        true_labels = np.unique(test_y)

        plt.imshow(cm, cmap=plt.cm.Blues)
        plt.colorbar()
        plt.xticks(np.arange(len(true_labels)), true_labels)
        plt.yticks(np.arange(len(true_labels)), true_labels)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion matrix')

        for i in range(len(true_labels)):
            for j in range(len(true_labels)):
                plt.text(j, i, cm[i, j], ha='center', va='center', color='red')

        # plt.show()
        plt.savefig(f"logs_images/cm-lda.jpg")

        # 清除当前 axes 上的所有图像元素
        plt.cla()
        # 或者，如果你想清除整个 Figure 包括所有子图，并重新开始
        plt.clf()

        # 计算FPR, TPR以及阈值
        fpr, tpr, thresholds = roc_curve(test_y, y_pred)

        # 绘制ROC曲线并标记AUC
        plt.figure(figsize=(8, 6), dpi=150)
        plt.plot(fpr, tpr, label=f'ROC curve (area = {auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')  # 纵横坐标轴对角线
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")

        # 保存图像
        plt.savefig(f"logs_images/ROC_curve_AUC-lda.png", bbox_inches='tight')
        exit()


    elif kind == "up":
        # 创建一个随机上采样器实例
        ros = RandomOverSampler()
        # 对少数类样本进行上采样
        train_x, train_y = ros.fit_resample(train_x, train_y)
    elif kind == "down":
        # 创建一个随机下采样器实例
        rus = RandomUnderSampler()
        # 对训练集中的多数类样本进行下采样
        train_x, train_y = rus.fit_resample(train_x, train_y)

    train_x = torch.tensor(train_x, dtype=torch.float32)
    test_x = torch.tensor(test_x, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.int64)
    test_y = torch.tensor(test_y, dtype=torch.int64)

    return train_x, train_y, test_x, test_y

def create_tensor(train_x, train_y, test_x, test_y, t=False):
    if kind not in ["up", "down"]:
        train_x, train_y = random_data(train_x, train_y)
        if t:
            test_x, test_y = random_data(test_x, test_y)

    train_data = TensorDataset(train_x, train_y)
    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = TensorDataset(test_x, test_y)
    test_data = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_data, test_data



def train():

    train_x, train_y, test_x, test_y = read_data()

    if name == "DNN":
        model = getattr(models, model_name)(num_lens)
    else:
        model = getattr(models, model_name)()

    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fc = nn.CrossEntropyLoss()

    loss_old = 100

    train_result = []
    test_result = []

    for epoch in range(1, epochs + 1):
        train_data, test_data = create_tensor(train_x, train_y, test_x, test_y)
        pbar = tqdm(train_data)
        loss_all = 0
        acc_all = 0
        for step, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            out = model(x)

            loss = loss_fc(out, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_all += loss.item()
            loss_time = loss_all / (step + 1)

            acc = torch.mean((y == torch.argmax(out, dim=-1)).float())

            acc_all += acc
            acc_time = acc_all / (step + 1)

            s = "train => epoch:{} - step:{} - loss:{:.4f} - loss_time:{:.4f} - acc:{:.4f} - acc_time:{:.4f}".format(
                epoch, step, loss, loss_time, acc, acc_time)
            pbar.set_description(s)

            train_result.append(s + "\n")

        with torch.no_grad():
            pbar = tqdm(test_data)
            loss_all = 0
            acc_all = 0
            for step, (x, y) in enumerate(pbar):
                x, y = x.to(device), y.to(device)
                out = model(x)

                loss = loss_fc(out, y)

                loss_all += loss.item()
                test_loss_time = loss_all / (step + 1)

                acc = torch.mean((y == torch.argmax(out, dim=-1)).float())

                acc_all += acc
                acc_time = acc_all / (step + 1)

                s = "test => epoch:{} - step:{} - loss:{:.4f} - loss_time:{:.4f} - acc:{:.4f} - acc_time:{:.4f}".format(
                    epoch, step, loss, test_loss_time, acc, acc_time)
                pbar.set_description(s)

                test_result.append(s + "\n")

        with open(f"logs/train_result-{name+kind}.txt", "w") as f:
            f.writelines(train_result)

        with open(f"logs/test_result-{name+kind}.txt", "w") as f:
            f.writelines(test_result)

        if loss_old > test_loss_time:
            loss_old = test_loss_time
            torch.save(model.state_dict(), f"weights/model-{name+kind}.pkl")

    ys = []
    prs = []

    model.load_state_dict(torch.load(f"weights/model-{name+kind}.pkl"))
    model.eval()

    train_data, test_data = create_tensor(train_x, train_y, test_x, test_y, False)

    with torch.no_grad():
        pbar = tqdm(test_data)
        for step, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            out = model(x)

            p = torch.argmax(out, dim=-1).cpu().numpy()
            y = y.cpu().numpy()

            ys.append(y)
            prs.append(p)

    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(prs, axis=0)

    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    print(name+kind)
    print(f"precision为：{precision * 100}%")
    print(f"recall为：{recall * 100}%")
    print(f"f1为：{f1 * 100}%")
    print(f"accuracy：{accuracy * 100}%")
    print(f"mcc：{mcc * 100}%")
    print(f"auc：{auc * 100}%")

    cm = confusion_matrix(y_true, y_pred)
    true_labels = np.unique(y_true)

    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xticks(np.arange(len(true_labels)), true_labels)
    plt.yticks(np.arange(len(true_labels)), true_labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion matrix')

    for i in range(len(true_labels)):
        for j in range(len(true_labels)):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='red')

    # plt.show()
    plt.savefig(f"logs_images/cm1-{name+kind}.jpg")

    # 清除当前 axes 上的所有图像元素
    plt.cla()
    # 或者，如果你想清除整个 Figure 包括所有子图，并重新开始
    plt.clf()

    # 计算FPR, TPR以及阈值
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # 绘制ROC曲线并标记AUC
    plt.figure(figsize=(8, 6), dpi=150)
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # 纵横坐标轴对角线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    # 保存图像
    plt.savefig(f"logs_images/ROC_curve_AUC-{name+kind}.png", bbox_inches='tight')


if __name__ == '__main__':
    train()