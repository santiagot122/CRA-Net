import torch
import torch.nn as nn
from utils import ConvBNAct


class DynamicScaleAwareHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super(DynamicScaleAwareHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Depthwise separable convolutions for lightweight feature processing
        self.dws_conv1 = ConvBNAct(in_channels, 2 * in_channels, kernel_size=3, depthwise=True)
        self.dws_conv2 = ConvBNAct(in_channels, 2 * in_channels, kernel_size=5, depthwise=True)

        # Adaptive average pooling to extract global feature vectors
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # SE Block for compensating the use of DWSConv
        self.se_block = nn.Sequential(
            nn.Linear(4 * in_channels, in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 8, 4 * in_channels),
            nn.ReLU(inplace=True),
        )

        # Convolution to refine the added features
        self.conv1x1 = ConvBNAct(2 * in_channels, 2 * in_channels, kernel_size=1)

        # Prediction layers
        self.classifier = nn.Conv2d(2 * in_channels, self.num_classes, kernel_size=1)
        self.regressor = nn.Conv2d(2 * in_channels, 4, kernel_size=1)

    def forward(self, x):
        # Step 1: Apply depthwise separable convolutions
        f1 = self.dws_conv1(x)
        f2 = self.dws_conv2(x)

        # Step 2: Concatenate the features from both streams
        f_concat = torch.cat([f1, f2], dim=1)

        # Step 3: Apply global average pooling
        f_pooled = self.avgpool(f_concat)
        f = f_pooled.view(f_pooled.size(0), -1)

        # Step 4: Transform features through SE Block
        f_transformed = self.se_block(f)

        # Step 5: Reshape the transformed features for broadcasting
        f_transformed = f_transformed.view(f_transformed.size(0), 4 * self.in_channels, 1, 1)

        # Step 6: Split transformed features and scale the original streams and add up
        f1_prime = f1 * f_transformed[:, : 2 * self.in_channels, :, :]
        f2_prime = f2 * f_transformed[:, 2 * self.in_channels :, :, :]
        f_prime = self.conv1x1(f1_prime + f2_prime)

        # Step 7: Generate predictions
        cls_logits = self.classifier(f_prime)
        bbox_preds = self.regressor(f_prime)

        # Reshape and permute classification and regression logits
        B, _, H, W = cls_logits.shape
        cls_logits = cls_logits.view(B, self.num_classes, H, W)
        cls_logits = cls_logits.permute(0, 2, 3, 1)  # (B, H, W, C)
        bbox_preds = bbox_preds.view(B, 4, H, W)
        bbox_preds = bbox_preds.permute(0, 2, 3, 1)  # (B, H, W, 4)

        return cls_logits, bbox_preds
