import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        depthwise: bool = False,
    ):
        super(ConvBNAct, self).__init__()
        padding = kernel_size // 2
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

        if depthwise:
            # Depthwise separable convolution
            self.depth_conv = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                padding=padding,
                groups=in_channels,
                bias=False,
            )
            self.point_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            # Standard convolution
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, padding=padding, bias=False
            )

    def forward(self, x):
        if hasattr(self, "depth_conv"):
            x = self.depth_conv(x)
            x = self.point_conv(x)
        else:
            x = self.conv(x)

        x = self.bn(x)
        x = self.activation(x)
        return x


class DetectionLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, focal_alpha=0.25, focal_gamma=2.0):
        super(DetectionLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def focal_loss(self, cls_logits, cls_targets):
        """Compute the focal loss between logits and the ground truth labels."""
        ce_loss = F.cross_entropy(cls_logits, cls_targets, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_weight = (1 - p_t) ** self.focal_gamma
        focal_loss = self.focal_alpha * focal_weight * ce_loss
        return focal_loss.mean()

    def smooth_l1_loss(self, bbox_preds, bbox_targets):
        """Compute the Smooth L1 loss for bounding box regression."""
        l1_loss = F.smooth_l1_loss(bbox_preds, bbox_targets, reduction="none")
        return l1_loss.mean()

    def forward(self, cls_logits, bbox_preds, cls_targets, bbox_targets):
        # Reshape logits and targets to ensure their dimensions are aligned for loss computation
        B, H, W, C = cls_logits.shape
        cls_logits = cls_logits.reshape(B * H * W, C)
        bbox_preds = bbox_preds.reshape(B * H * W, 4)

        cls_targets = cls_targets.reshape(
            B * H * W,
        )
        bbox_targets = bbox_targets.reshape(B * H * W, 4)

        # Compute classification and bounding box regression losses
        cls_loss = self.focal_loss(cls_logits, cls_targets)
        bbox_loss = self.smooth_l1_loss(bbox_preds, bbox_targets)

        # Weighted sum of classification and bounding box regression losses
        total_loss = self.alpha * cls_loss + self.beta * bbox_loss

        return total_loss
