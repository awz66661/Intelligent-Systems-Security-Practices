import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image 
import random
from torchvision import transforms
from torch.utils.data import DataLoader

def fit_shape(single_x, device):
    # 单张图片 -> model.forward()
    return (
        torch.from_numpy(single_x).unsqueeze(0).permute(0, 3, 1, 2).float().to(device) / 255
    )
    
def load_cifar10():
    batch_size = 128
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),])

    train_data = torchvision.datasets.CIFAR10(root='data/', train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    print('Clean train data is prepared.')

    test_data = torchvision.datasets.CIFAR10(root='data/', train=False, transform=transform, download=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    print('Clean test data is prepared.')
    
    return train_data, test_data, train_loader, test_loader


class BackdoorMNIST(torchvision.datasets.MNIST):
    '''
    这里的处理思路与PoisonMNIST类相同，我们将预处理和投毒步骤放在初始化函数中完成
    '''
    def __init__(self, root, ratio, target=3, size=3, trigger=None, *args, **kwargs):
        super(BackdoorMNIST, self).__init__(root, *args, **kwargs)
        self.ratio = ratio  # 浮点数，表示数据集的投毒比例
        self.size = size  # 整数，表示trigger小方块的边长
        self.target = target  # 整数，表示贴上trigger的样本指向的目标标签

        data_float = []
        for idx in range(self.__len__()):
            img = self.data[idx]
            img = Image.fromarray(img.numpy(), mode="L")
            if self.transform is not None:
                img = self.transform(img)
            data_float.append(img)
        self.data = torch.stack(data_float, dim=0)

        def apply_global_trigger(x: torch.Tensor, trigger: torch.Tensor):
            if len(x.shape) == 4:
                x_t = x + trigger.unsqueeze(0).repeat([x.shape[0], 1, 1, 1])
            elif len(x.shape) == 3:
                x_t = x + trigger
            else:
                raise NotImplementedError
            x_t = torch.clamp(x_t, min=0., max=1.)
            return x_t

        # 接下来我们在数据集中同时对图片和标签做投毒
        for idx in range(self.__len__()):
            if random.random() < ratio:
                img = self.data[idx]
                if trigger is not None:
                    img = apply_global_trigger(img, trigger)
                else:
                    H, W = img.shape[1], img.shape[2]
                    img[:, H - self.size : H, W - self.size : W] = 1.
                self.data[idx] = img
                self.targets[idx] = self.target
    
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        return img, target