import torch
import torch.nn as nn
from resnet_backbone import ResNetBackbone
from hsfem_module import HighLevelSemanticFeatureExtractionModule
from lfem_module import LowLevelFeatureEnhancementModule
from dsh_module import DynamicScaleAwareHead


class DetectionModel(nn.Module):
    def __init__(self, num_classes: int):
        super(DetectionModel, self).__init__()

        # Backbone for feature extraction (ResNet-like architecture)
        self.backbone = ResNetBackbone()

        # High-Level Feature Attention Module (HFAM) to process the deepest feature map
        self.hfam_p4 = HighLevelSemanticFeatureExtractionModule(in_channels=2048, out_channels=256)

        # Low-Level Feature Composition Modules (LFCM) for fusing high-level and lower-resolution feature maps
        self.lfcm_p3 = LowLevelFeatureEnhancementModule(
            high_in_channels=2048, low_in_channels=1024, out_channels=256
        )
        self.lfcm_p2 = LowLevelFeatureEnhancementModule(
            high_in_channels=1024, low_in_channels=512, out_channels=256
        )
        self.lfcm_p1 = LowLevelFeatureEnhancementModule(
            high_in_channels=512, low_in_channels=256, out_channels=256
        )

        # Detection head to generate predictions for classification and bounding boxes
        self.detection_head = DynamicScaleAwareHead(in_channels=256, num_classes=num_classes)

    def forward(self, x):
        # Step 1: Extract multi-scale features using the backbone
        c4, c3, c2, c1 = self.backbone(x)

        # Step 2: Process the deepest feature map using HFAM
        p4 = self.hfam_p4(c4)

        # Step 3: Fuse high- and low-level features using LFCM modules
        p3 = self.lfcm_p3(c4, c3)
        p2 = self.lfcm_p2(c3, c2)
        p1 = self.lfcm_p1(c2, c1)

        # Step 4: Generate the predictions
        cls_logits4, bbox_preds4 = self.detection_head(p4)
        cls_logits3, bbox_preds3 = self.detection_head(p3)
        cls_logits2, bbox_preds2 = self.detection_head(p2)
        cls_logits1, bbox_preds1 = self.detection_head(p1)

        cls_logits_list = [cls_logits4, cls_logits3, cls_logits2, cls_logits1]
        bbox_preds_list = [bbox_preds4, bbox_preds3, bbox_preds2, bbox_preds1]

        return cls_logits_list, bbox_preds_list
