import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder, MNIST
from torchvision import transforms
from torch import autograd
from torch.autograd import Variable
from torchvision.utils import make_grid

import numpy as np
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.label_emb = nn.Embedding(10, 10)

        self.model = nn.Sequential(
            nn.Linear(110, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z, labels):
        z = z.view(z.size(0), 100)
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out.view(x.size(0), 28, 28)

generator = Generator().to(device)

generator = generator.load_state_dict(torch.load('generator_state.pth'))
for i in range(10):
    z = Variable(torch.randn(100, 100)).cuda()
    labels = torch.LongTensor([i for i in range(10) for _ in range(10)]).cuda()
    images = generator(z, labels).unsqueeze(1)
    generated_img = make_grid(images)
    save_generator_image(generated_img, f"/home/jorgecarranzapena01/csci-4353-JorgeCaPe/labs/lab24/img/outputted_img{i + 1}.png")
