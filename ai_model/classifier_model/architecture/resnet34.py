import torch.nn as nn
import torch
from image_preprocess import dataset

class ResBlock_Tier2(nn.Module):
    def __init__(self, in_channels, intermediate_channels):
        super(ResBlock_Tier2, self).__init__()

        # Layers
        # To make the residual input equal size as output channel.

        downsample = False
        self.skip_connection = nn.Sequential()  # Default

        if intermediate_channels == 2 * in_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    intermediate_channels,
                    kernel_size=1,
                    stride=2,
                    bias=False,
                ),
                nn.BatchNorm2d(intermediate_channels),
            )
            downsample = True

        # Downsampling the output shape.
        if downsample:
            self.conv1 = nn.Conv2d(
                in_channels,
                intermediate_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )

        else:
            self.conv1 = nn.Conv2d(
                in_channels,
                intermediate_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )

        self.bn1 = nn.BatchNorm2d(intermediate_channels)

        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Residual to be added later.
        identity = self.skip_connection(x)

        # ---------------------------
        # Layer-1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # ----------------------------

        # ---------------------------
        # Layer-2
        x = self.conv2(x)
        x = self.bn2(x)
        x += identity
        x = self.relu(x)
        # ----------------------------

        return x



class ResNet_Tier2(nn.Module):
    def __init__(self, img_channels, num_layers, num_classes, dropout_rate=0.5):
        super(ResNet_Tier2, self).__init__()
        self.dropout_rate = dropout_rate
        # Layers
        # Layer-0 Output shape : 64 X 56 X 56
        self.layer0 = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Residual Blocks layers
        self.layer1 = self._make_layer(ResBlock_Tier2, num_layers[0], 64, 64)
        self.layer2 = self._make_layer(ResBlock_Tier2, num_layers[1], 64, 128)
        self.layer3 = self._make_layer(ResBlock_Tier2, num_layers[2], 128, 256)
        self.layer4 = self._make_layer(ResBlock_Tier2, num_layers[3], 256, 512)
        self.dropout = nn.Dropout(self.dropout_rate)
        # FC Layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def _make_layer(
        self, ResBlock_Tier2, num_residual_blocks, in_channels, intermediate_channels
    ):
        layers = []
        only_once = True

        for i in range(num_residual_blocks):
            layers.append(ResBlock_Tier2(in_channels, intermediate_channels))
            if only_once:
                in_channels = intermediate_channels
                only_once = False

        return nn.Sequential(*layers)


# Tier-2 Resnets.
def ResNet34(img_channel=3, num_classes=len(dataset.classes)):
    return ResNet_Tier2(img_channel, [3,4,6,3], num_classes)

