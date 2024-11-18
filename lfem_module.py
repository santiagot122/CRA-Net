import torch
import torch.nn as nn
from utils import ConvBNAct


class LowLevelFeatureEnhancementModule(nn.Module):
    def __init__(self, high_in_channels: int, low_in_channels: int, out_channels: int):
        super(LowLevelFeatureEnhancementModule, self).__init__()

        # Upsample for matching the spatial dimension of the high-level features
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Convolution for efficient processing of combined low- and high-level features
        self.conv1x1 = ConvBNAct(low_in_channels + high_in_channels, out_channels, kernel_size=1)

        # Adaptive Average Pooling to extract global feature vector
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Final convolution layer to consolidate the refined features
        self.final_conv = ConvBNAct(out_channels + low_in_channels, out_channels, kernel_size=3)

    def forward(self, high_feature, low_feature):
        # Step 1: Upsample the high-level feature map to match the resolution of the low-level feature map
        upsampled_high = self.upsample(high_feature)

        # Step 2: Concatenate low- and high-level feature maps along the channel dimension
        concat_feat = torch.cat([low_feature, upsampled_high], dim=1)

        # Step 3: Process the concatenated feature map using convolution
        enhanced_feature = self.conv1x1(concat_feat)

        # Step 4: Extract global context using adaptive average pooling
        global_context = self.avgpool(enhanced_feature)

        # Step 5: Add the global context back to the enhanced feature map (broadcasting is applied)
        enhanced_feature = enhanced_feature + global_context.expand_as(enhanced_feature)

        # Step 6: Combine enhanced feature with low level feature through concatenation
        residual_feature = torch.cat([enhanced_feature, low_feature], dim=1)

        # Step 7: Apply the final convolution to consolidate the residual connected features
        output_feature = self.final_conv(residual_feature)

        return output_feature
