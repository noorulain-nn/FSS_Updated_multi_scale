import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def load_backbone(backbone_name: str) -> tuple[nn.Module, int]:
    """
    Classification backbone (used by main.py only).
    Freezes all layers except the last block, strips classifier.
    """
    name = backbone_name.lower().strip()

    def _pretrained(ctor):
        try:
            return ctor(weights="IMAGENET1K_V1")
        except Exception:
            return ctor(pretrained=True)

    if name in {"resnet18", "resnet34", "resnet50", "resnet101"}:
        ctor = {"resnet18": models.resnet18, "resnet34": models.resnet34,
                "resnet50": models.resnet50, "resnet101": models.resnet101}[name]
        m = _pretrained(ctor)
        for n, p in m.named_parameters():
            if not n.startswith("layer4."):
                p.requires_grad = False
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
        backbone = nn.Sequential(
            *(c for c in [m.conv1, m.bn1, m.relu, m.maxpool,
                          m.layer1, m.layer2, m.layer3, m.layer4]),
            m.avgpool,
            nn.Flatten(1),
        )
        return backbone, feat_dim

    if name == "inception_v3":
        m = _pretrained(models.inception_v3)
        m.aux_logits = False
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
        for n, p in m.named_parameters():
            if "Mixed_7" not in n:
                p.requires_grad = False
        backbone = nn.Sequential(
            m.Conv2d_1a_3x3, m.Conv2d_2a_3x3, m.Conv2d_2b_3x3, m.maxpool1,
            m.Conv2d_3b_1x1, m.Conv2d_4a_3x3, m.maxpool2,
            m.Mixed_5b, m.Mixed_5c, m.Mixed_5d,
            m.Mixed_6a, m.Mixed_6b, m.Mixed_6c, m.Mixed_6d, m.Mixed_6e,
            m.Mixed_7a, m.Mixed_7b, m.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
        )
        return backbone, feat_dim

    if name == "squeezenet1_0":
        m = _pretrained(models.squeezenet1_0)
        for n, p in m.named_parameters():
            if "features.12" not in n:
                p.requires_grad = False
        backbone = nn.Sequential(m.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(1))
        return backbone, 512

    if name == "squeezenet1_1":
        m = _pretrained(models.squeezenet1_1)
        for n, p in m.named_parameters():
            if "features.12" not in n:
                p.requires_grad = False
        backbone = nn.Sequential(m.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(1))
        return backbone, 512

    if name in {"densenet121", "densenet161", "densenet169"}:
        ctor = {"densenet121": models.densenet121,
                "densenet161": models.densenet161,
                "densenet169": models.densenet169}[name]
        m = _pretrained(ctor)
        for n, p in m.named_parameters():
            if not n.startswith("features.denseblock4"):
                p.requires_grad = False
        feat_dim = m.classifier.in_features
        m.classifier = nn.Identity()
        backbone = nn.Sequential(
            m.features, nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(1)
        )
        return backbone, feat_dim

    if name in {"vgg16", "vgg19"}:
        ctor = {"vgg16": models.vgg16, "vgg19": models.vgg19}[name]
        m = _pretrained(ctor)
        cutoff = 24 if name == "vgg16" else 27
        for idx, layer in m.features.named_children():
            if int(idx) < cutoff:
                for p in layer.parameters():
                    p.requires_grad = False
        m.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        m.classifier = nn.Identity()
        backbone = nn.Sequential(m.features, m.avgpool, nn.Flatten(1))
        return backbone, 512

    raise ValueError(f"Unsupported backbone_name: {backbone_name!r}")


# ─────────────────────────────────────────────────────────────────
# SEGMENTATION BACKBONE  (used by main_seg.py + APM.py)
# ─────────────────────────────────────────────────────────────────

def load_backbone_seg(backbone_name: str):
    """
    Segmentation backbone: raw ResNet module (no pooling/flatten).
    encode() in SegAPM calls individual layer attributes directly.

    Issue 6 fix:
      Previously trained BOTH layer3 and layer4. The APM paper trains
      only the LAST block. For few-shot segmentation on a small support
      set, training two blocks increases the risk of overfitting to
      the support images. Freezing layer3 and training only layer4
      matches the APM paper protocol and reduces that risk.

      If you find val_IoU plateaus early you can re-enable layer3 by
      changing the startswith check back to ("layer4.", "layer3.").
    """
    name = backbone_name.lower().strip()

    def _pretrained(ctor):
        try:
            return ctor(weights="IMAGENET1K_V1")
        except Exception:
            return ctor(pretrained=True)

    if name in {"resnet18", "resnet34", "resnet50", "resnet101"}:
        ctor = {"resnet18": models.resnet18, "resnet34": models.resnet34,
                "resnet50": models.resnet50, "resnet101": models.resnet101}[name]
        m = _pretrained(ctor)

        # Issue 6 fix: train ONLY layer4 (was layer3 + layer4)
        for n, p in m.named_parameters():
            if not n.startswith("layer4."):
                p.requires_grad = False

        return m, 2048

    raise ValueError(f"Unsupported backbone_name: {backbone_name!r}")


# ─────────────────────────────────────────────────────────────────
# IMPROVED FPN DECODER
# ─────────────────────────────────────────────────────────────────

class ImprovedFPNDecoder(nn.Module):
    """
    FPN-style decoder: fuses layer4 (coarse) and layer3 (fine) features,
    upsamples to 473x473, outputs 2-channel logits (bg + fg).
    Unchanged from original.
    """

    def __init__(self):
        super().__init__()
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, kernel_size=1),   # 2 channels: bg + fg
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, feats4_reduced, feats3_reduced):
        f4     = self.conv4(feats4_reduced)
        f4     = self.upsample(f4)
        f3     = self.conv3(feats3_reduced)
        fused  = f4 + f3
        x      = self.refine(fused)
        x      = self.upsample(x)
        x      = self.upsample(x)
        logits = F.interpolate(x, size=(473, 473), mode='bilinear', align_corners=True)
        return logits
