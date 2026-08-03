"""4-class CXR classifier for template-based report output (Normal, COVID, Viral, Lung_Opacity)."""

import torch
import torch.nn as nn
from torchvision.models import resnet18


class CXRClassifier(nn.Module):
    """ResNet-18 backbone + linear head for 4-class CXR classification."""

    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        rn = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.conv1 = rn.conv1
        self.bn1 = rn.bn1
        self.relu = rn.relu
        self.maxpool = rn.maxpool
        self.layer1 = rn.layer1
        self.layer2 = rn.layer2
        self.layer3 = rn.layer3
        self.layer4 = rn.layer4
        self.avgpool = rn.avgpool
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
