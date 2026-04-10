import torch
import torch.nn as nn
import torch.nn.functional as F
from Models import ImprovedFPNDecoder   # ← Updated import

class MemoryModuleFSS(nn.Module):
    """
    APM memory module adapted for binary Few-Shot Segmentation.
    """
    def __init__(self, num_classes: int, feature_dim: int):
        super().__init__()
        self.register_buffer('memory', torch.zeros(num_classes, feature_dim))
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.initialized = [False] * num_classes

    def extract_prototype(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_ds = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=features.shape[-2:],
            mode='nearest'
        )
        masked = features * mask_ds
        n_pixels = mask_ds.sum() + 1e-6
        proto = masked.sum(dim=[2, 3]) / n_pixels
        return F.normalize(proto.squeeze(0), p=2, dim=0)

    def forward(self, query_features: torch.Tensor) -> torch.Tensor:
        B, C, H, W = query_features.shape
        q_norm = F.normalize(query_features, p=2, dim=1)
        m_norm = F.normalize(self.memory, p=2, dim=1)
        q_flat = q_norm.permute(0, 2, 3, 1).reshape(B, H * W, C)
        sim = torch.matmul(q_flat, m_norm.t())
        return sim.reshape(B, H, W, self.num_classes).permute(0, 3, 1, 2)

    def update_memory(self, features: torch.Tensor, mask: torch.Tensor, class_label: int) -> None:
        with torch.no_grad():
            new_proto = self.extract_prototype(features, mask.to(features.device))
            if not self.initialized[class_label]:
                self.memory[class_label] = new_proto
                self.initialized[class_label] = True
                return

            sim = F.cosine_similarity(new_proto.unsqueeze(0), self.memory[class_label].unsqueeze(0), dim=1).item()
            alpha = 1.0 - sim
            self.memory[class_label] = (1.0 - alpha) * self.memory[class_label] + alpha * new_proto


class SegAPM(nn.Module):
    """
    APM-FSS with Improved Decoder + Better Contextual Prior
    """
    def __init__(self, backbone, num_classes, feature_dim, output_size=(473, 473)):
        super().__init__()
        self.backbone = backbone
        self.memory_module = MemoryModuleFSS(num_classes, feature_dim)
        self.output_size = output_size
        
        # Use stronger decoder
        self.decoder = ImprovedFPNDecoder()
        
        # Channel projection: 2048/1024 → 256
        self.proj4 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.proj3 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

    def encode(self, imgs: torch.Tensor):
        x = self.backbone.conv1(imgs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        feats3 = self.backbone.layer3(x)
        feats4 = self.backbone.layer4(feats3)
        
        feats4_reduced = self.proj4(feats4)
        feats3_reduced = self.proj3(feats3)
        
        return feats4_reduced, feats3_reduced, feats4   # reduced for decoder, raw for memory

    def forward(self, imgs: torch.Tensor):
        feats4_reduced, feats3_reduced, feats4_raw = self.encode(imgs)
        
        # Memory similarity (pixel-wise)
        similarity_map = self.memory_module(feats4_raw)   # [B, 1, H', W']
        
        # Stronger contextual prior
        prior = F.interpolate(similarity_map, size=(473, 473), mode='bilinear', align_corners=True)
        prior = torch.sigmoid(prior) * 0.7                # scaled to avoid overpowering decoder
        
        # Decoder prediction
        seg_logits = self.decoder(feats4_reduced, feats3_reduced)
        
        # Combine (additive prior)
        seg_logits = seg_logits + prior
        
        return seg_logits, feats4_raw, similarity_map
