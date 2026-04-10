import torch
import torch.nn as nn
import torch.nn.functional as F
from Models import ImprovedFPNDecoder


class MemoryModuleFSS(nn.Module):
    """
    APM memory module adapted for binary Few-Shot Segmentation.

    FIXES:
      Issue 1 — num_classes is now always 2 (bg slot 0 + fg slot 1).
                The decoder also outputs 2 channels, so memory and decoder
                are consistent. Previously num_classes=1 left background
                with no prototype.
      Issue 2 — prototype is re-normalised onto the unit hypersphere after
                every EMA blend. Previously the stored vector drifted off
                the sphere, making cosine_similarity values unreliable on
                the next update call.
    """

    def __init__(self, num_classes: int, feature_dim: int):
        super().__init__()
        # Issue 1 fix: always 2 slots regardless of what caller passes.
        # Slot 0 = background, Slot 1 = foreground.
        self.num_classes = 2
        self.feature_dim = feature_dim
        self.register_buffer('memory', torch.zeros(self.num_classes, feature_dim))
        self.initialized = [False] * self.num_classes

    def extract_prototype(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Masked average pooling: mean of feature vectors at foreground pixels.

        Args:
            features : (1, C, H_f, W_f)  backbone spatial features
            mask     : (H, W)             binary mask on original image resolution

        Returns:
            proto : (C,) L2-normalised prototype
        """
        mask_ds = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=features.shape[-2:],
            mode='nearest'
        )                                              # (1, 1, H_f, W_f)
        masked    = features * mask_ds                 # (1, C, H_f, W_f)
        n_pixels  = mask_ds.sum() + 1e-6
        proto     = masked.sum(dim=[2, 3]) / n_pixels  # (1, C)
        return F.normalize(proto.squeeze(0), p=2, dim=0)   # (C,)

    def forward(self, query_features: torch.Tensor) -> torch.Tensor:
        """
        Pixel-wise cosine similarity between every spatial location and
        both (bg + fg) memory prototypes.

        Args:
            query_features : (B, C, H_f, W_f)

        Returns:
            sim_map : (B, 2, H_f, W_f)  — used as additive prior to decoder logits
        """
        B, C, H, W = query_features.shape
        q_norm  = F.normalize(query_features, p=2, dim=1)              # (B, C, H, W)
        m_norm  = F.normalize(self.memory, p=2, dim=1)                 # (2, C)
        q_flat  = q_norm.permute(0, 2, 3, 1).reshape(B, H * W, C)     # (B, HW, C)
        sim     = torch.matmul(q_flat, m_norm.t())                     # (B, HW, 2)
        return sim.reshape(B, H, W, self.num_classes).permute(0, 3, 1, 2)  # (B, 2, H, W)

    def update_memory(self, features: torch.Tensor, mask: torch.Tensor, class_label: int) -> None:
        """
        Adaptive EMA update for one class prototype.

        Issue 2 fix: after the EMA blend, the result is re-normalised back
        onto the unit hypersphere with F.normalize. Without this the stored
        vector drifts so that the cosine_similarity check on the next call
        uses a non-unit vector, making alpha unreliable.

        Args:
            features    : (1, C, H_f, W_f)
            mask        : (H, W) binary mask (CPU or GPU — handled internally)
            class_label : 0 for background, 1 for foreground
        """
        with torch.no_grad():
            new_proto = self.extract_prototype(features, mask.to(features.device))

            if not self.initialized[class_label]:
                self.memory[class_label] = new_proto
                self.initialized[class_label] = True
                return

            sim   = F.cosine_similarity(
                new_proto.unsqueeze(0),
                self.memory[class_label].unsqueeze(0),
                dim=1
            ).item()
            alpha = 1.0 - sim
            blended = (1.0 - alpha) * self.memory[class_label] + alpha * new_proto

            # Issue 2 fix: re-normalise after blend
            self.memory[class_label] = F.normalize(blended, p=2, dim=0)


class SegAPM(nn.Module):
    """
    APM-FSS with Improved FPN Decoder + Corrected Memory Prior.

    FIXES:
      Issue 1 — MemoryModuleFSS always has 2 slots; decoder outputs 2 channels;
                both background and foreground prototypes are maintained.
      Issue 3 — prior scaling changed from 0.25 to 1.0 (no suppression).
                The memory similarity map is in [-1, 1]; after sigmoid it is
                [0.27, 0.73]. Multiplying by 0.25 compressed it to [0.07, 0.18]
                which is ~3-6% of typical decoder logit magnitude — effectively
                zero influence. At scale 1.0 the prior meaningfully shifts the
                final logits toward the memory's prediction.
    """

    PRIOR_SCALE = 1.0   # Issue 3 fix: was 0.25

    def __init__(self, backbone, num_classes, feature_dim, output_size=(473, 473)):
        super().__init__()
        self.backbone      = backbone
        # Issue 1 fix: always 2 classes internally
        self.memory_module = MemoryModuleFSS(num_classes=2, feature_dim=feature_dim)
        self.output_size   = output_size
        self.decoder       = ImprovedFPNDecoder()

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
        feats3        = self.backbone.layer3(x)
        feats4        = self.backbone.layer4(feats3)
        feats4_reduced = self.proj4(feats4)
        feats3_reduced = self.proj3(feats3)
        return feats4_reduced, feats3_reduced, feats4

    def forward(self, imgs: torch.Tensor):
        feats4_reduced, feats3_reduced, feats4_raw = self.encode(imgs)

        # Memory similarity: (B, 2, H_f, W_f) — both bg and fg channels
        similarity_map = self.memory_module(feats4_raw)

        # Issue 3 fix: prior at full scale (was * 0.25, now * 1.0)
        prior = F.interpolate(
            similarity_map, size=(473, 473),
            mode='bilinear', align_corners=True
        )
        prior = torch.sigmoid(prior) * self.PRIOR_SCALE

        # Decoder prediction
        seg_logits = self.decoder(feats4_reduced, feats3_reduced)   # (B, 2, 473, 473)

        # Additive prior
        seg_logits = seg_logits + prior

        return seg_logits, feats4_raw, similarity_map
