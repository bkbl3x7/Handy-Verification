import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class HandNetSmall(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = self.pool(x).view(x.size(0), -1)
        x = F.normalize(self.fc(x))
        return x


def make_backbone(name: str, embed_dim=128):
    if name == "handnet":
        return HandNetSmall(embed_dim)
    if name == "resnet18":
        m = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        in_f = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Linear(in_f, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )

        class ResnetWrap(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                z = self.m(x)
                return F.normalize(z)

        return ResnetWrap(m)
    raise ValueError("Unknown backbone")


class LinearClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

