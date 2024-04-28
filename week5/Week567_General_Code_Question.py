import os

import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

import pickle
from matplotlib import pyplot as plt


########## IMPLEMENT THE CODE BELOW, COMMENT OUT IRRELEVENT CODE IF NEEDED ##########
##### Model Definition #####
# TODO: Week 5, Task 1
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 使用nn.Conv2d和nn.Linear定义模型的结构
        # 输入: 1*28*28 -> Conv1: 6*28*28 -> MaxPool1: 6*14*14 -> Conv2: 16*10*10 -> MaxPool2: 16*5*5 -> FC1: 120 -> FC2: 84 -> FC3: 10
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)  # 1个输入通道，6个输出通道，卷积核大小为5，填充为2
        self.conv2 = nn.Conv2d(6, 16, 5)  # 6个输入通道，16个输出通道，卷积核大小为5
        self.pool = nn.MaxPool2d(2, 2)  # 最大池化，池化核大小为2，步长为2
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 全连接层，输入大小为16*5*5，输出大小为120
        self.fc2 = nn.Linear(120, 84)  # 全连接层，输入大小为120，输出大小为84
        self.fc3 = nn.Linear(84, 10)  # 全连接层，输入大小为84，输出大小为10
    
    def forward(self, x):
        # 定义前向传播过程
        # 输入形状: [Batch, 1, 28, 28]
        x = self.pool(torch.relu(self.conv1(x)))  # 应用卷积、ReLU激活和最大池化
        x = self.pool(torch.relu(self.conv2(x)))  # 应用卷积、ReLU激活和最大池化
        x = x.view(-1, 16 * 5 * 5)  # 重塑张量形状
        x = torch.relu(self.fc1(x))  # 应用全连接层和ReLU激活
        x = torch.relu(self.fc2(x))  # 应用全连接层和ReLU激活
        x = self.fc3(x)  # 应用全连接层
        o = x  # 输出
        return o


##### Model Evaluation #####
# TODO: Week 5, Task 2
def evaluate(imgs, labels, model):
    # TODO：用model预测imgs，并得到预测标签pred_label
    pred_label = None
    model.eval()
    with torch.no_grad():
        pred_label = model(imgs)
        pred_label = torch.argmax(pred_label, dim=1)
    

    # TODO：计算预测标签pred_label与真实标签labels的匹配数目
    correct_cnt = None
    correct_cnt = torch.sum(pred_label == labels).item()

    
    print(f'match rate: {correct_cnt/labels.shape[0]}')
    return pred_label


##### Adversarial Attacks #####
# TODO: Week 5, Task 2
def fgsm(imgs, epsilon, model, criterion, labels):
    model.eval()

    # 将输入转换为浮点型张量
    adv_xs = imgs.float()
    # 设置requires_grad为True，以便计算梯度
    adv_xs.requires_grad = True

    # TODO：模型前向传播，计算loss，然后loss反传
    o = model(adv_xs)  # 模型前向传播，得到输出
    loss = criterion(o, labels)  # 计算损失
    loss.backward()  # 反向传播，计算梯度

    # TODO：得到输入的梯度、生成对抗样本
    grad = adv_xs.grad  # 获取输入的梯度
    adv_xs = adv_xs + epsilon * torch.sign(grad)  # 生成对抗样本

    # TODO：对扰动做截断，保证对抗样本的像素值在合理域内
    adv_xs = torch.clamp(adv_xs, 0, 1)  # 对扰动进行截断，确保像素值在0和1之间
    model.train()

    return adv_xs.detach()


# TODO: Week 6, Task 1
def pgd(imgs, epsilon, iter, model, criterion, labels):
    model.eval()

    adv_xs = imgs.float()

    for i in range(iter):
        # Forward and compute loss, then backward

        # Retrieve grad and generate adversarial example, note to detach

        # Clip perturbation
        pass
    
    model.train()

    return adv_xs.detach()


# TODO: Week 6, Task 2
def fgsm_target(imgs, epsilon, model, criterion, labels):
    model.eval()

    adv_xs = imgs.float()
    adv_xs.requires_grad = True

    # Forward and compute loss, then backward

    # Retrieve grad and generate adversarial example, note to detach
    # Note to compute TARGETED loss and the sign of the perturbation

    # Clip perturbation

    model.train()

    return adv_xs.detach()

# TODO: Week 6, Task 2
def pgd_target(imgs, epsilon, iter, model, criterion, labels):
    model.eval()

    adv_xs = imgs.float()

    for i in range(iter):
        adv_xs.requires_grad = True
        
        # Forward and compute loss, then backward

        # Retrieve grad and generate adversarial example, note to detach
        # Note to compute TARGETED loss and the sign of the perturbation

        # Clip perturbation
    
    model.train()

    return adv_xs.detach()


########## NO NEED TO MODIFY CODE BELOW ##########
##### Data Loader #####
def load_mnist(batch_size):
    if not os.path.exists('data/'):
        os.mkdir('data/')

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_set = torchvision.datasets.MNIST(root='data/', transform=transform, train=True, download=True)
    test_set = torchvision.datasets.MNIST(root='data/', transform=transform, train=False, download=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


##### Visualization #####
def visualize_benign(imgs, labels):
    fig = plt.figure(figsize=(8, 7))
    for idx, (img, label) in enumerate(zip(imgs, labels)):
        ax = fig.add_subplot(4, 5, idx + 1)
        ax.imshow(img[0], cmap='gray')
        ax.set_title(f'label: {label.item()}')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def visualize_adv(imgs, true_labels, pred_labels):
    fig = plt.figure(figsize=(8, 8))
    for idx, (img, true_label, pred_label) in enumerate(zip(imgs, true_labels, pred_labels)):
        ax = fig.add_subplot(4, 5, idx + 1)
        ax.imshow(img[0], cmap='gray')
        ax.set_title(f'true label: {true_label.item()}\npred label: {pred_label.item()}')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def visualize_target_adv(imgs, target_labels, pred_labels):
    fig = plt.figure(figsize=(8, 8))
    for idx, (img, true_label, pred_label) in enumerate(zip(imgs, target_labels, pred_labels)):
        ax = fig.add_subplot(4, 5, idx + 1)
        ax.imshow(img[0], cmap='gray')
        ax.set_title(f'target label: {true_label.item()}\npred label: {pred_label.item()}')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
