import torch
import torch.nn as nn
import torchvision.models as models


class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        resnet = models.resnet50(pretrained=True)

        # Disable inplace=True for all ReLU activations in the resnet model
        for module in resnet.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

        # Split ResNet model into layers to output multi-scale features
        self.layer1 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
        )  # C1
        self.layer2 = resnet.layer2  # C2
        self.layer3 = resnet.layer3  # C3
        self.layer4 = resnet.layer4  # C4

    def forward(self, x):
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return [c4, c3, c2, c1]  # Return feature maps in descending spatial resolution
