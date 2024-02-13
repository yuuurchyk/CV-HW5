import torch.nn as nn
import torchvision


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.encoder = torchvision.models.vgg16_bn()
        self.linear = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.encoder(x)
        x = self.linear(x)
        return x
