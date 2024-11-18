import torch
import torch.nn as nn
from utils import ConvBNAct


class HighLevelSemanticFeatureExtractionModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(HighLevelSemanticFeatureExtractionModule, self).__init__()

        # Reduce the number of channels for computational efficiency
        self.reduce = ConvBNAct(in_channels, out_channels, kernel_size=3)

        # Multi-scale feature extraction using convolutions with different kernel sizes
        self.conv1 = ConvBNAct(out_channels, out_channels, kernel_size=1)
        self.conv3 = ConvBNAct(out_channels, out_channels, kernel_size=3)
        self.conv5 = ConvBNAct(out_channels, out_channels, kernel_size=5)
        self.conv7 = ConvBNAct(out_channels, out_channels, kernel_size=7)

        # Restore channels after combining multi-scale features
        self.restore_channel = ConvBNAct(4 * out_channels, out_channels, kernel_size=3)
        self.another_conv = ConvBNAct(out_channels, out_channels, kernel_size=3)

        # Spatial attention layer
        self.conv1x1 = nn.Conv2d(out_channels, 1, kernel_size=1, padding=0)
        # Channel attention layer
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        # Refine attention-enhanced features
        self.conv3x3 = ConvBNAct(out_channels, out_channels, kernel_size=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Step 1: Reduce the input channels
        x = self.reduce(x)  # Reduced feature map: (batch_size, out_channels, H, W)

        # Step 2: Extract multi-scale features
        c1 = self.conv1(x)
        c3 = self.conv3(x)
        c5 = self.conv5(x)
        c7 = self.conv7(x)

        # Step 3: Combine the multi-scale features
        concatfeat = torch.cat([c1, c3, c5, c7], dim=1)
        featmap = self.restore_channel(concatfeat)
        featmap = self.another_conv(featmap)

        # Step 4: Compute spatial and channel attention
        spmap = self.conv1x1(featmap)
        chmap = self.maxpool(featmap)
        spchmap = spmap * chmap

        # Step 5: Refine attention-enhanced features
        raw_attn = self.conv3x3(spchmap)
        attn_map = self.sigmoid(raw_attn)

        # Step 6: Apply attention map to the input features with residual connection
        final_output = attn_map * x + x

        return final_output
