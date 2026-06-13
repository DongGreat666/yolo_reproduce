# 机构：人工智能研究所
# 人员：东
# 时间：2026/6/13 12:15
import torch
from torch import nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky_relu(self.bn(self.conv(x)))


class YOLOv1(nn.Module):
    def __init__(self, in_channels=3, split_size=7, num_boxes=2, num_classes=20):
        super().__init__()

        self.S = split_size
        self.B = num_boxes
        self.C = num_classes

        self.darknet = nn.Sequential(
            CNNBlock(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2),

            CNNBlock(64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            CNNBlock(192, 128, kernel_size=1, stride=1, padding=0),
            CNNBlock(128, 256, kernel_size=3, stride=1, padding=1),
            CNNBlock(256, 256, kernel_size=1, stride=1, padding=0),
            CNNBlock(256, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            CNNBlock(512, 256, kernel_size=1, stride=1, padding=0),
            CNNBlock(256, 512, kernel_size=3, stride=1, padding=1),
            CNNBlock(512, 256, kernel_size=1, stride=1, padding=0),
            CNNBlock(256, 512, kernel_size=3, stride=1, padding=1),
            CNNBlock(512, 256, kernel_size=1, stride=1, padding=0),
            CNNBlock(256, 512, kernel_size=3, stride=1, padding=1),
            CNNBlock(512, 256, kernel_size=1, stride=1, padding=0),
            CNNBlock(256, 512, kernel_size=3, stride=1, padding=1),

            CNNBlock(512, 512, kernel_size=1, stride=1, padding=0),
            CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0),
            CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1),
            CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0),
            CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1),

            CNNBlock(1024, 1024, kernel_size=3, stride=1, padding=1),
            CNNBlock(1024, 1024, kernel_size=3, stride=2, padding=1),

            CNNBlock(1024, 1024, kernel_size=3, stride=1, padding=1),
            CNNBlock(1024, 1024, kernel_size=3, stride=1, padding=1),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * self.S * self.S, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, self.S * self.S * (self.C + self.B * 5))
        )

    def forward(self, x):
        x = self.darknet(x)
        x = self.fc(x)
        x = x.view(-1, self.S, self.S, self.C + self.B * 5)
        return x


if __name__ == "__main__":
    model = YOLOv1()
    x = torch.randn(2, 3, 448, 448)
    y = model(x)
    print(y.shape)







