import torch
from torch import nn
from model.yolov1 import CNNBlock

class ResNetYOLO(nn.Module):
    def __init__(
        self,
        split_size=7,
        num_boxes=2,
        num_classes=20,
        pretrained=True,
        detection_head="fc_head",
    ):
        super().__init__()

        self.S = split_size
        self.B = num_boxes
        self.C = num_classes
        self.detection_head = detection_head
        self.output_size = self.C + self.B * 5

        self.backbone = self._build_backbone(pretrained)
        self.pool = nn.AdaptiveAvgPool2d((self.S, self.S))

        if detection_head == "fc_head":
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512 * self.S * self.S, 4096),
                nn.Dropout(0.5),
                nn.LeakyReLU(0.1),
                nn.Linear(
                    4096,
                    self.S * self.S * self.output_size,
                ),
            )

        elif detection_head == "cnn_head":
            self.head = nn.Sequential(
                CNNBlock(
                    512,
                    1024,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                CNNBlock(
                    1024,
                    1024,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.Conv2d(
                    1024,
                    self.output_size,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
            )

        else:
            raise ValueError(
                f"Unknown detection_head: {detection_head}"
            )

    @staticmethod
    def _build_backbone(pretrained):
        try:
            from torchvision.models import ResNet34_Weights, resnet34

            weights = ResNet34_Weights.DEFAULT if pretrained else None
            model = resnet34(weights=weights)
        except ImportError:
            raise RuntimeError(
                "torchvision is required for the ResNet34 backbone."
            )
        except TypeError:
            from torchvision.models import resnet34

            model = resnet34(pretrained=pretrained)

        return nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)  # [N, 512, 7, 7]

        if self.detection_head == "fc_head":
            x = self.head(x)
            return x.view(
                -1,
                self.S,
                self.S,
                self.output_size,
            )

        x = self.head(x)                 # [N, 30, 7, 7]
        x = x.permute(0, 2, 3, 1)       # [N, 7, 7, 30]
        return x.contiguous()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.head.parameters():
            param.requires_grad = True

    def unfreeze_layer4(self):
        self.freeze_backbone()

        for param in self.backbone[7].parameters():
            param.requires_grad = True

    def unfreeze_layer3_layer4(self):
        self.freeze_backbone()

        for param in self.backbone[6].parameters():
            param.requires_grad = True

        for param in self.backbone[7].parameters():
            param.requires_grad = True

    def unfreeze_backbone(self):
        for param in self.parameters():
            param.requires_grad = True

if __name__ == "__main__":
    x = torch.randn(2, 3, 448, 448)

    fc_model = ResNetYOLO(
        pretrained=False,
        detection_head="fc_head",
    )
    print("FC:", fc_model(x).shape)

    cnn_model = ResNetYOLO(
        pretrained=False,
        detection_head="cnn_head",
    )
    print("CNN:", cnn_model(x).shape)
