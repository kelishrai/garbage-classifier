import torch.nn as nn
import torch

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, intermediate_channels, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(intermediate_channels, intermediate_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(intermediate_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, img_channels, num_classes=12, dropout_rate=0.5):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.dropout_rate = dropout_rate

        self.layer0 = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(BottleneckBlock, 64, 3)
        self.layer2 = self._make_layer(BottleneckBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(BottleneckBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(BottleneckBlock, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(self.dropout_rate)  # Add dropout layer
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)  # Apply dropout
        x = self.fc(x)

        return x

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

def ResNet50():
    return ResNet(img_channels=3, num_classes=12)