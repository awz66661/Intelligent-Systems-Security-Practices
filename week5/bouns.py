import os
import numpy as np
import pickle
from matplotlib import pyplot as plt

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from Week567_General_Code_Question import load_mnist

batch_size = 128
train_loader, test_loader = load_mnist(batch_size=batch_size)
mnist_dim = 28 * 28
z_dim = 64

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# 定义判别器模型
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G = Generator(z_dim, mnist_dim).to(device)
D = Discriminator(mnist_dim).to(device)
print(G, D)
    
lr = 0.02
criterion = nn.BCEWithLogitsLoss()
G_optimizer = torch.optim.Adam(G.parameters(), lr=lr)
D_optimizer = torch.optim.Adam(D.parameters(), lr=lr)
G_loss_history = []
D_loss_history = []

def show():
    G.eval()
    G_gen = G(torch.randn((10, z_dim), device=device))
    imgs = G_gen.detach().cpu().data.numpy().reshape(G_gen.shape[0], 28, 28)

    fig = plt.figure(figsize=(8, 3))
    for idx, img in enumerate(imgs):
        ax = fig.add_subplot(2, 5, idx + 1)
        ax.imshow(img, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

epochs = 100
for e in range(epochs):
    D_loss_avg = 0.0
    G_loss_avg = 0.0
    
    for real_images, _ in tqdm(train_loader, desc='Epoch %d' % e):
        batch_size = real_images.size(0)
        real_images = real_images.view(-1, mnist_dim).to(device)
        
        # Train Discriminator
        D_optimizer.zero_grad()
        
        z = torch.randn(batch_size, z_dim).to(device)
        fake_images = G(z)

        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        real_loss = criterion(D(real_images), real_labels)
        fake_loss = criterion(D(fake_images.detach()), fake_labels)  # 使用detach()避免计算生成器的梯度
        D_loss = real_loss + fake_loss

        D_loss.backward()
        D_optimizer.step()

        # Train Generator
        G_optimizer.zero_grad()

        G_loss = criterion(D(fake_images), real_labels)

        G_loss.backward()
        G_optimizer.step()

        D_loss_avg += D_loss.item() * batch_size
        G_loss_avg += G_loss.item() * batch_size
    
    D_loss_avg /= len(train_loader.dataset)
    G_loss_avg /= len(train_loader.dataset)
    
    print('epoch %d: D_loss %.3f, G_loss %.3f' % (e, D_loss_avg, G_loss_avg))
    D_loss_history.append(D_loss_avg)
    G_loss_history.append(G_loss_avg)

    if e % 10 == 0:
        show()

show()

fig, ax = plt.subplots()
plt.plot(np.array(D_loss_history), label='Discriminator')
plt.plot(np.array(G_loss_history), label='Generator')
plt.title("Training Losses")
plt.legend()
plt.show()