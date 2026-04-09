import torch
import torch.nn as nn
import torch.nn.functional as F
# from Models import SimpleMultiScaleDecoder
from Models import MSDNetStyleDecoder
# Memory Module

class MemoryModule(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super(MemoryModule, self).__init__()
        self.memory = nn.Parameter(torch.randn(num_classes, feature_dim), requires_grad=False)  # Memory matrix
        self.memory_labels = [i for i in range(5)] #[-1] * num_classes  # Placeholder for memory-class associations
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.memory, mean=0, std=0.01)
        
    def forward(self, features):
        
        normalized_features = F.normalize(features, p=2, dim=1)  # Shape: [batch_size, feature_dim]
        normalized_memory = F.normalize(self.memory, p=2, dim=1)  # Shape: [num_classes, feature_dim]
    
        # Compute cosine similarity scores between features and memory slots
        # Resulting shape: [batch_size, num_classes]
        
        scores = torch.matmul(normalized_features, normalized_memory.t())
        logits = scores 
        
        predicted_slot = torch.argmax(logits, dim=1)
        predicted_label = [self.memory_labels[slot.item()] for slot in predicted_slot]
        return logits, predicted_slot, predicted_label

    

    # Memory Update Mechanism
    def update_memory(self, features, attention_scores, true_label, predicted_slot):
        
        update_count = 0
        
        # If this is the first time the true label is encountered
        if true_label not in self.memory_labels:

            # If the highest attention slot is available, assign the label to this slot
            if self.memory_labels[predicted_slot] == -1:
                self.memory_labels[predicted_slot] = true_label
                true_slot = self.memory_labels.index(true_label)
                self.memory.data[true_slot] = (
                    F.normalize(features, p=2, dim=-1)
                )
                return 0

            else:
                true_slot = self.memory_labels.index(-1)
                self.memory_labels[true_slot] = true_label
                self.memory.data[true_slot] = F.normalize(features, p=2, dim=-1)
                
                return 1

        true_slot = self.memory_labels.index(true_label)
        #"""
        similarity = F.cosine_similarity(F.normalize(features, p=2, dim=-1), F.normalize(self.memory.data[true_slot], p=2, dim=-1), dim=-1)
        adaptive_lr = 1 - similarity
        self.memory.data[true_slot] = (
            (1 - adaptive_lr) * self.memory.data[true_slot]
            + (adaptive_lr * F.normalize(features, p=2, dim=-1))
        )#"""

        
        return 0       
 
        
# Full Model
class MemoryEnabledCNN(nn.Module):
    def __init__(self, backbone, num_classes, feature_dim):
        super(MemoryEnabledCNN, self).__init__()
        self.backbone = backbone
        self.memory_module = MemoryModule(num_classes, feature_dim)

    def forward(self, x):
        features = self.backbone(x)
        attention_scores, predicted_slot, predicted_label = self.memory_module(features)
        return predicted_label, features, attention_scores, predicted_slot


# ─────────────────────────────────────────────────────────────────
# FSS MEMORY MODULE  (added for FSS)
# ─────────────────────────────────────────────────────────────────

class MemoryModuleFSS(nn.Module):
    """
    APM memory module adapted for binary Few-Shot Segmentation.

    Three changes from the classification MemoryModule:
      1. Prototype = Masked Average Pooling (MAP) of spatial feature map,
         not a raw GAP feature vector.
         Why: prototype must represent the foreground REGION only.
         Reference: PANet (Wang et al., ICCV 2019) introduces MAP for FSS.

      2. Similarity is computed per pixel, not per image.
         Why: output must be a 2D map [H', W'], not a scalar per image.

      3. update_memory() takes a spatial feature map + binary mask,
         extracts MAP prototype, then applies the same EMA rule as
         the original APM paper (alpha = 1 - cosine_sim).
    """

    def __init__(self, num_classes: int, feature_dim: int):
        super().__init__()
        # Non-trainable prototype storage — same design as original APM
        self.register_buffer(
            'memory',
            torch.zeros(num_classes, feature_dim)
        )
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.initialized = [False] * num_classes

    # ── 1. Prototype extraction via Masked Average Pooling ──────
    def extract_prototype(self,
                          features: torch.Tensor,
                          mask:     torch.Tensor) -> torch.Tensor:
        """
        Computes class prototype by averaging features over
        the foreground region defined by the binary mask.

        Args:
            features : [1, C, H', W']  spatial feature map from encoder
            mask     : [H, W]          binary mask at original image size
                                       (1 = foreground, 0 = background)
        Returns:
            prototype : [C]  L2-normalised prototype vector
        """
        # Resize mask from image resolution down to feature map resolution
        mask_ds = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),  # [1, 1, H, W]
            size=features.shape[-2:],                 # → [1, 1, H', W']
            mode='nearest'
        )

        # Average features only over foreground pixels
        # numerator:   sum of feature vectors where mask=1
        # denominator: count of foreground pixels
        masked   = features * mask_ds                 # [1, C, H', W']
        n_pixels = mask_ds.sum() + 1e-6
        proto    = masked.sum(dim=[2, 3]) / n_pixels  # [1, C]

        return F.normalize(proto.squeeze(0), p=2, dim=0)  # [C]

    # ── 2. Forward: pixel-level cosine similarity ───────────────
    def forward(self, query_features: torch.Tensor) -> torch.Tensor:
        """
        Computes cosine similarity between every spatial location in
        the query feature map and every stored prototype.

        This replaces the image-level dot product in classification APM.

        Args:
            query_features : [B, C, H', W']
        Returns:
            similarity_map : [B, num_classes, H', W']
                             positive values = more similar to prototype
        """
        B, C, H, W = query_features.shape

        # L2-normalise along channel dim — same as original APM
        q_norm = F.normalize(query_features, p=2, dim=1)  # [B, C, H', W']
        m_norm = F.normalize(self.memory,    p=2, dim=1)  # [K, C]

        # Reshape for batched matrix multiply
        # q_flat : [B, H'*W', C]
        q_flat = q_norm.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # sim : [B, H'*W', K]
        sim = torch.matmul(q_flat, m_norm.t())

        # Reshape to [B, K, H', W']
        return sim.reshape(B, H, W, self.num_classes).permute(0, 3, 1, 2)

    # ── 3. Memory update — same adaptive EMA as APM paper ───────
    def update_memory(self,
                      features:    torch.Tensor,
                      mask:        torch.Tensor,
                      class_label: int) -> None:
        """
        Updates stored prototype using the same adaptive EMA rule
        as the original APM paper (Eq. 13-14):

            alpha  = 1 - cosine_sim(new_proto, stored_proto)
            memory = (1 - alpha) * old_proto + alpha * new_proto

        The only difference from classification APM is that new_proto
        is computed via MAP (masked average pooling) instead of being
        a raw GAP feature vector.

        Args:
            features    : [1, C, H', W']  spatial feature map (detached)
            mask        : [H, W]          binary ground-truth mask (CPU)
            class_label : int             which memory slot to update
        """
        with torch.no_grad():
            new_proto = self.extract_prototype(features, mask.to(features.device))

            # First time this class is seen — direct write (same as APM)
            if not self.initialized[class_label]:
                self.memory[class_label] = new_proto
                self.initialized[class_label] = True
                return

            # Adaptive EMA — identical to APM Eq. 13-14
            sim   = F.cosine_similarity(
                        new_proto.unsqueeze(0),
                        self.memory[class_label].unsqueeze(0),
                        dim=1
                    ).item()
            alpha = 1.0 - sim   # high novelty → large update, low novelty → small

            self.memory[class_label] = (
                (1.0 - alpha) * self.memory[class_label]
                +       alpha * new_proto
            )


class SegAPM(nn.Module):
    """
    APM-FSS with MULTI-SCALE decoder
    """
    def __init__(self, backbone, num_classes, feature_dim, output_size=(473, 473)):
        super().__init__()
        self.backbone = backbone
        self.memory_module = MemoryModuleFSS(num_classes, feature_dim)
        self.output_size = output_size
        
        self.decoder = MSDNetStyleDecoder()
        
        # === THIS IS THE IMPORTANT PART WE ADD ===
        # Reduce 2048 channels → 256 channels (what decoder wants)
        self.proj4 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.proj3 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, bias=False),  # layer3 has 1024 channels
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

    def encode(self, imgs: torch.Tensor):
        """Extract features from backbone"""
        x = self.backbone.conv1(imgs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        feats3 = self.backbone.layer3(x)   # [B, 1024, ...]
        feats4 = self.backbone.layer4(feats3)  # [B, 2048, ...]
        
        # Reduce channels so decoder is happy
        feats4_reduced = self.proj4(feats4)
        feats3_reduced = self.proj3(feats3)
        
        return feats4_reduced, feats3_reduced, feats4   # return reduced + original feats4

    def forward(self, imgs: torch.Tensor):
        # Get features
        feats4_reduced, feats3_reduced, feats4_raw = self.encode(imgs)
        
        # Memory module uses the rich (2048) features - better for prototypes
        similarity_map = self.memory_module(feats4_raw)
        
        # Contextual prior
        contextual_prior = F.sigmoid(similarity_map)
        contextual_prior = F.interpolate(contextual_prior, size=(473,473), mode='bilinear', align_corners=True)
        
        # Decoder uses the reduced (256) features
        seg_logits = self.decoder(feats4_reduced, feats3_reduced)
        
        # Add prior
        seg_logits = seg_logits + contextual_prior
        
        return seg_logits, feats4_raw, similarity_map
