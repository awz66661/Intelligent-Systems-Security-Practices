import os

import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

import pickle
from matplotlib import pyplot as plt


########## IMPLEMENT THE CODE BELOW, COMMENT OUT IRRELEVENT CODE IF NEEDED ##########
##### Model Definition #####
# TODO: Week 5, Task 1 (请迁移Week 5已实现代码)
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # TODO：在这里定义模型的结构。主要依赖于nn.Conv2d和nn.Linear
        # 1*28*28 -> 6*28*28 -> 6*14*14 -> 16*10*10 -> 16*5*5 -> 120 -> 84 -> 10
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)  
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        # TODO：在这里定义前向传播过程。这里输入的x形状是[Batch, 1, 28, 28]
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        o = x
        return o


##### Model Evaluation #####
# TODO: Week 5, Task 2 (请迁移Week 5已实现代码)
def evaluate(imgs, labels, model):
    # # TODO：用model预测imgs，并得到预测标签pred_label
    # pred_label = None

    # # TODO：计算预测标签pred_label与真实标签labels的匹配数目
    # correct_cnt = None
    
    # print(f'match rate: {correct_cnt/labels.shape[0]}')
    # return pred_label

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

# TODO: Week 7, Task 1
def evaluate_dataloader(dataloader, model):
    model.eval()
    
    correct_cnt, sample_cnt = 0, 0

    t = tqdm(dataloader)
    for img, label in t:
        # TODO: Predict label for img, update correct_cnt, sample_cnt
        pred_label = model(img)
        pred_label = torch.argmax(pred_label, dim=1)
        correct_cnt += torch.sum(pred_label == label).item()
        sample_cnt += label.shape[0]
        t.set_postfix(test_acc=correct_cnt/sample_cnt)


##### Adversarial Attacks #####
# TODO: Week 5, Task 2 (请迁移Week 5已实现代码)
def fgsm(imgs, epsilon, model, criterion, labels):
    model.eval()

    adv_xs = imgs.float()

    adv_xs.requires_grad = True

    # TODO：模型前向传播，计算loss，然后loss反传

    #o = model(adv_xs)
    o = model(adv_xs)
    loss = criterion(o, labels)
    loss.backward()

    # TODO：得到输入的梯度、生成对抗样本
    grad = adv_xs.grad
    adv_xs = adv_xs + epsilon * torch.sign(grad)

    # TODO：对扰动做截断，保证对抗样本的像素值在合理域内
    adv_xs = torch.clamp(adv_xs, 0, 1)
    model.train()

    return adv_xs.detach()


# TODO: Week 6, Task 1 (请迁移Week 6已实现代码)
# def pgd(imgs, epsilon, iter, model, criterion, labels):
#     model.eval()
#     alphe = 0.07

#     adv_xs = imgs.float()
#     xs = adv_xs.clone()
  
#     # 和fgsm类似，只是需要多次迭代

#     for i in range(iter):
#         # Forward and compute loss, then backward 
#         adv_xs.requires_grad = True
#         o = model(adv_xs)
#         model.zero_grad()
#         loss = criterion(o, labels)
#         loss.backward()
#         # Retrieve grad and generate adversarial example, note to detach

#         with torch.no_grad():

#             dadv_xs = adv_xs.grad.sign()
#             adv_xs += alphe * dadv_xs
#             dlt = torch.clamp(adv_xs - xs, min=-epsilon, max=epsilon)
#             adv_xs = xs + dlt
#             adv_xs = torch.clamp(adv_xs, min=0, max = 1.)
#             adv_xs = adv_xs.detach()
#             # Clip perturbation
        
    
#     model.train()

#     return adv_xs.detach()
# def pgd(imgs, epsilon, iter, model, criterion, labels):
#     model.eval()
#     alpha = 0.07

#     adv_xs = imgs.float().clone()
#     xs = imgs.float().clone()
  
#     for i in range(iter):
#         adv_xs.requires_grad = True
#         outputs = model(adv_xs)
#         loss = criterion(outputs, labels)
#         model.zero_grad()
#         loss.backward()

#         # 更新对抗样本，使用 no_grad 优化计算
#         with torch.no_grad():
#             adv_xs += alpha * adv_xs.grad.sign()  # 应用扰动
#             adv_xs = torch.clamp(adv_xs, 0, 1)  # 确保有效图像范围
#             # 限制扰动在 epsilon 范围内
#             adv_xs = torch.clamp(adv_xs, xs - epsilon, xs + epsilon)

#     model.train()
#     return adv_xs

def pgd(imgs, epsilon, iter, model, criterion, labels):
    model.eval()
    alpha = 0.07
    #print(imgs._version)
    adv_xs = imgs.clone().detach()  # Ensures that adv_xs is a separate copy
    xs = imgs.clone().detach()
    #print(imgs._version)
    for i in range(iter):
        adv_xs.requires_grad = True
        outputs = model(adv_xs)
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            adv_xs += alpha * adv_xs.grad.sign()
            adv_xs = torch.clamp(adv_xs, min=0, max=1)
            adv_xs = torch.clamp(adv_xs, xs - epsilon, xs + epsilon)
            adv_xs = adv_xs.detach()  # Detach to ensure no gradients are accumulated

    model.train()
    return adv_xs




# TODO: Week 6, Task 2 (请迁移Week 6已实现代码)
def fgsm_target(imgs, epsilon, model, criterion, labels):
    model.eval()
    alphe = 0.07
    adv_xs = imgs.float()
    adv_xs.requires_grad = True

    # Forward and compute loss, then backward
    o = model(adv_xs)
    loss = criterion(o, labels)
    loss.backward()

    # Retrieve grad and generate adversarial example, note to detach
    # Note to compute TARGETED loss and the sign of the perturbation
    dadv_xs = -adv_xs.grad.sign()
    adv_xs = adv_xs + alphe * dadv_xs
    adv_xs = adv_xs.detach()
    # Clip perturbation
    adv_xs = torch.clamp(adv_xs, 0, 1)
    model.train()

    return adv_xs.detach()

# TODO: Week 6, Task 2 (请迁移Week 6已实现代码)
def pgd_target(imgs, epsilon, iter, model, criterion, labels):
    model.eval()

    adv_xs = imgs.float()
    xs = adv_xs.clone()
    for i in range(iter):
        adv_xs.requires_grad = True
        
        # Forward and compute loss, then backward
        o = model(adv_xs)
        loss = criterion(o, labels)
        loss.backward()
        
        # Retrieve grad and generate adversarial example, note to detach
        # Note to compute TARGETED loss and the sign of the perturbation
        dadv_xs = adv_xs.grad.sign()
        dlt = adv_xs - xs + epsilon * dadv_xs
        adv_xs = adv_xs.detach()
        dlt = torch.clamp(dlt, min=-epsilon, max=epsilon)
        adv_xs -= dlt
        adv_xs = torch.clamp(adv_xs, min=0, max=1)
        adv_xs = adv_xs.detach()
        # Clip perturbation
    
    model.train()

    return adv_xs.detach()


# TODO: Week 6, Bonus (请迁移Week 6已实现代码，或注释)
def nes(imgs, epsilon, model, labels, sigma, n):
    """
    labels: ground truth labels
    sigma: search variance
    n: number of samples used for estimation for each img
    """
    model.eval()

    adv_xs = imgs.reshape(-1, 28 * 28).float()

    grad = torch.zeros_like(adv_xs)
    # TODO: Estimate gradient for each sample adv_x in adv_xs

    adv_xs = adv_xs.detach() - epsilon * grad.sign()
    adv_xs = torch.clamp(adv_xs, min=0., max=1.)

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
