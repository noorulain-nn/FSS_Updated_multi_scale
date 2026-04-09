"""
APM Few-Shot Segmentation — main_seg.py

Methodology matches APM classification exactly:
  - Support images used for BOTH loss computation AND memory update
  - Query images used for evaluation only (no gradient)
  - Same hyperparameters as APM paper Table 3
  - Same 5-episode structure with seeds 42/142/242/342/442
  - Only last backbone block fine-tuned
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import random

import Data_Loader
import Models
import APM
import PLOT
import warnings
warnings.filterwarnings('ignore', category=SyntaxWarning)
# ── Device ──────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# ── Config — same as APM paper Table 3 ─────────────────────────
DATA_ROOT    = "./data/fss-data"  
FOLD         = 0                       # Pascal-5i fold 0-3
K_SHOT       = 1                       # 1-shot (or 5 for 5-shot)
IMG_SIZE     = 473
NUM_CLASSES  = 1                       # binary FSS: 1 foreground class
BATCH_SIZE   = 6
NUM_EPOCHS   = 10                      # per APM paper Table 3
LR           = 0.0005                  # per APM paper Table 3
RANDOM_SEEDS = [42, 142, 242, 342, 442]# per APM paper Table 3
BACKBONE     = "resnet50"


# ── Metric: mean-IoU ────────────────────────────────────────────
def compute_iou(pred_mask: torch.Tensor,
                true_mask: torch.Tensor) -> float:
    """
    pred_mask : [H, W] binary tensor (0 or 1)
    true_mask : [H, W] binary tensor (0 or 1)
    Returns foreground IoU.
    """
    pred  = pred_mask.bool()
    true  = true_mask.bool()
    inter = (pred & true).sum().float()
    union = (pred | true).sum().float()
    return (inter / (union + 1e-6)).item()


# ── Validation function ─────────────────────────────────────────
def validate(model, val_loader, criterion):
    """
    Uses support images to populate memory,
    then evaluates on query images.
    No gradient computation anywhere.
    """
    model.eval()
    total_loss = 0.0
    total_iou  = 0.0
    count      = 0

    with torch.no_grad():
        for s_imgs, s_masks, q_imgs, q_masks in val_loader:
            B = s_imgs.shape[0]

            # ① Reset memory for this episode
            model.memory_module.initialized = [False] * NUM_CLASSES

            # ② Populate memory from support images (no gradient)
                        # ② Populate memory from support images (no gradient)
            for b in range(B):
                img  = s_imgs[b, 0].unsqueeze(0).to(device)  # [1, 3, H, W]
                mask = s_masks[b, 0]                          # [H, W]  CPU ok
                
                # NEW: encode now returns two feature maps (feats4, feats3)
                feats4_reduced, feats3_reduced, feats4_raw = model.encode(img)            # feats4 for memory, feats3 for decoder
                # print(f"Debug - feats4: {feats4.shape} | feats3: {feats3.shape}")   
                feat = feats4                                 # keep variable name 'feat' for update_memory
                
                model.memory_module.update_memory(
                    feat, mask, class_label=0
                )

            # ③ Evaluate on query images
            q_imgs  = q_imgs.to(device)
            q_masks = q_masks.long().to(device)               # [B, H, W]

            seg_logits, _, _ = model(q_imgs)                  # [B, 2, H, W]
            loss = criterion(seg_logits, q_masks)

            total_loss += loss.item() * B
            pred = seg_logits.argmax(dim=1)                   # [B, H, W]
            for b in range(B):
                total_iou += compute_iou(pred[b].cpu(), q_masks[b].cpu())
            count += B

    return total_loss / count, total_iou / count


# ── Training function ───────────────────────────────────────────
def train(model, train_loader, val_loader,
          criterion, optimizer, scheduler,
          num_epochs, episode):
    """
    Mirrors APM classification training exactly:

      Same image (support) is used for:
        1. Forward pass → loss → backprop (fine-tunes backbone layer4)
        2. Memory prototype update (adaptive EMA, no gradient)

    Query images are never touched during training.
    """
    train_losses, val_losses = [], []
    train_ious,   val_ious   = [], []

    for epoch in range(num_epochs):
        model.train()
        run_loss  = 0.0
        run_iou   = 0.0
        count     = 0

        for s_imgs, s_masks, q_imgs, q_masks in train_loader:
            # s_imgs  : [B, k_shot, 3, H, W]
            # s_masks : [B, k_shot, H, W]
            # q_imgs, q_masks are loaded but NOT used in training
            # (only used during validate/test — same as APM classification)

            B = s_imgs.shape[0]

            # Take first shot only → [B, 3, H, W] and [B, H, W]
            support_img  = s_imgs[:, 0].to(device)    # [B, 3, H, W]
            support_mask = s_masks[:, 0]               # [B, H, W]  keep CPU for update

            optimizer.zero_grad()

            # ① Forward pass on support image
            #    Gradient flows through backbone layer4 — this is fine-tuning
            seg_logits, features, _ = model(support_img)   # [B, 2, H, W]

            # ② Pixel-level loss on support mask
            #    Equivalent to CE loss on support labels in classification APM
            loss = criterion(seg_logits, support_mask.long().to(device))
            loss.backward()
            optimizer.step()

            # ③ Memory update on same support image features
            #    Same adaptive EMA rule: alpha = 1 - cosine_sim
            #    Equivalent to update_memory() in classification APM
            with torch.no_grad():
                for b in range(B):
                    model.memory_module.update_memory(
                        features[b:b+1].detach(),   # [1, C, H', W']
                        support_mask[b],             # [H, W]
                        class_label=0
                    )

            # ── Batch metrics ──────────────────────────────────
            pred = seg_logits.argmax(dim=1)             # [B, H, W]
            run_loss += loss.item() * B
            for b in range(B):
                run_iou += compute_iou(
                    pred[b].cpu(),
                    support_mask[b].cpu()
                )
            count += B

        # ── Epoch summary ──────────────────────────────────────
        epoch_loss = run_loss / count
        epoch_iou  = run_iou  / count
        val_loss, val_iou = validate(model, val_loader, criterion)

        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        train_ious.append(epoch_iou)
        val_ious.append(val_iou)

        print(f"Episode {episode+1} | Epoch [{epoch+1}/{num_epochs}] "
              f"train_loss={epoch_loss:.4f}  train_IoU={epoch_iou:.4f}  "
              f"val_loss={val_loss:.4f}  val_IoU={val_iou:.4f}")

        scheduler.step()

    # Plots per episode (same as APM classification)
    PLOT.plot_bias_variance_curve(train_losses, val_losses)
    PLOT.plot_accuracy(train_ious, val_ious)

    return float(np.mean(val_ious))


# ── Test function ───────────────────────────────────────────────
def test(model, test_loader, criterion):
    """
    Populates memory from support, evaluates on query.
    No gradient. Returns mean-IoU over all test episodes.
    """
    model.eval()
    total_loss = 0.0
    total_iou  = 0.0
    count      = 0

    with torch.no_grad():
        for s_imgs, s_masks, q_imgs, q_masks in test_loader:
            B = s_imgs.shape[0]

            # Reset memory
            model.memory_module.initialized = [False] * NUM_CLASSES

            # Populate from support
                        
            for b in range(B):
                img  = s_imgs[b, 0].unsqueeze(0).to(device)
                mask = s_masks[b, 0]
                
                # NEW: encode now returns two feature maps
                feats4, feats3 = model.encode(img)
                feat = feats4                                 # use feats4 for memory update
                
                model.memory_module.update_memory(
                    feat, mask, class_label=0
                )

            # Evaluate on query
            q_imgs  = q_imgs.to(device)
            q_masks = q_masks.long().to(device)

            seg_logits, _, _ = model(q_imgs)
            loss = criterion(seg_logits, q_masks)

            total_loss += loss.item() * B
            pred = seg_logits.argmax(dim=1)
            for b in range(B):
                total_iou += compute_iou(pred[b].cpu(), q_masks[b].cpu())
            count += B

    test_loss = total_loss / count
    test_iou  = total_iou  / count
    print(f"Test  loss={test_loss:.4f}  mean-IoU={test_iou:.4f}")
    return test_iou


# ── Episode loop — same 5-episode structure as APM paper ────────
# ── Replace the episode loop at the bottom of main_seg.py ───────

# if __name__ == '__main__':

#     print("=" * 70)
#     print("APM Few-Shot Segmentation — 4 folds × 5 episodes each")
#     print("=" * 70)

#     criterion = nn.CrossEntropyLoss()

#     # Store results per fold
#     all_fold_val_ious  = []
#     all_fold_test_ious = []

#     for fold in range(4):   # ← outer loop over all 4 folds

#         print(f"\n{'#'*70}")
#         print(f"FOLD {fold}/3")
#         print(f"{'#'*70}")

#         fold_val_ious  = []
#         fold_test_ious = []

#         for ep_idx, seed in enumerate(RANDOM_SEEDS):

#             print(f"\n{'='*70}")
#             print(f"FOLD {fold} | EPISODE {ep_idx+1}/5 | seed={seed}")
#             print(f"{'='*70}\n")

#             torch.manual_seed(seed)
#             np.random.seed(seed)
#             random.seed(seed)

#             # Load data for this fold and seed
#             train_loader, val_loader, test_loader, _ = \
#                 Data_Loader.prepare_pascal5i(
#                     DATA_ROOT,
#                     fold=fold,          # ← changes each outer iteration
#                     k_shot=K_SHOT,
#                     img_size=IMG_SIZE,
#                     batch_size=BATCH_SIZE,
#                     seed=seed
#                 )

#             backbone, feat_dim = Models.load_backbone_seg(BACKBONE)
#             model = APM.SegAPM(
#                 backbone,
#                 num_classes=NUM_CLASSES,
#                 feature_dim=feat_dim,
#                 output_size=(IMG_SIZE, IMG_SIZE)
#             ).to(device)

#             optimizer = optim.Adam(model.parameters(), lr=LR)
#             scheduler = StepLR(optimizer, step_size=1, gamma=0.30)

#             val_iou  = train(model, train_loader, val_loader,
#                               criterion, optimizer, scheduler,
#                               NUM_EPOCHS, ep_idx)
#             test_iou = test(model, test_loader, criterion)

#             fold_val_ious.append(val_iou)
#             fold_test_ious.append(test_iou)

#             print(f"Fold {fold} Episode {ep_idx+1}: "
#                   f"val_IoU={val_iou:.4f}  test_IoU={test_iou:.4f}")

#         # Per-fold summary (averaged over 5 episodes)
#         fold_val_mean  = float(np.mean(fold_val_ious))
#         fold_test_mean = float(np.mean(fold_test_ious))
#         all_fold_val_ious.append(fold_val_mean)
#         all_fold_test_ious.append(fold_test_mean)

#         print(f"\nFold {fold} result: "
#               f"val_IoU={fold_val_mean:.4f}  test_IoU={fold_test_mean:.4f}")

#     # Final result — mean over all 4 folds (this is the number you report)
#     print(f"\n{'='*70}")
#     print("FINAL RESULTS — mean over all 4 folds × 5 episodes")
#     print(f"{'='*70}")
#     for f in range(4):
#         print(f"  Fold {f}: val={all_fold_val_ious[f]:.4f}  "
#               f"test={all_fold_test_ious[f]:.4f}")
#     print(f"\nOverall val  mean-IoU: {np.mean(all_fold_val_ious):.4f} "
#           f"± {np.std(all_fold_val_ious):.4f}")
#     print(f"Overall test mean-IoU: {np.mean(all_fold_test_ious):.4f} "
#           f"± {np.std(all_fold_test_ious):.4f}")
#     print(f"\nConfig: Pascal-5i, {K_SHOT}-shot, backbone={BACKBONE}")
#     print("=" * 70)
# ── Episode loop — ONLY FOLD 0 + 5 episodes (Recommended for remaining time) ─────
if __name__ == '__main__':

    print("=" * 70)
    print("APM Few-Shot Segmentation — FOLD 0 ONLY × 5 episodes")
    print("=" * 70)

    # criterion = nn.CrossEntropyLoss()
    class DiceLoss(nn.Module):
      def forward(self, pred, target):
          pred = torch.sigmoid(pred[:, 1])          # foreground channel
          target = (target == 1).float()
          intersection = (pred * target).sum()
          return 1 - (2 * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)

    criterion = DiceLoss()

    fold = 0                                      # ← Only Fold 0
    print(f"\n{'#'*70}")
    print(f"RUNNING ONLY FOLD {fold}")
    print(f"{'#'*70}")

    fold_val_ious  = []
    fold_test_ious = []

    for ep_idx, seed in enumerate(RANDOM_SEEDS):   # This will run all 5 seeds

        print(f"\n{'='*70}")
        print(f"FOLD {fold} | EPISODE {ep_idx+1}/5 | seed={seed}")
        print(f"{'='*70}\n")

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Load data for Fold 0
        train_loader, val_loader, test_loader, _ = \
            Data_Loader.prepare_pascal5i(
                DATA_ROOT,
                fold=fold,
                k_shot=K_SHOT,
                img_size=IMG_SIZE,
                batch_size=BATCH_SIZE,
                seed=seed
            )

        backbone, feat_dim = Models.load_backbone_seg(BACKBONE)
        model = APM.SegAPM(
            backbone,
            num_classes=NUM_CLASSES,
            feature_dim=feat_dim,
            output_size=(IMG_SIZE, IMG_SIZE)
        ).to(device)

        # ←←← Better optimizer for less overfitting (Recommended)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.30)

        val_iou  = train(model, train_loader, val_loader,
                         criterion, optimizer, scheduler,
                         NUM_EPOCHS, ep_idx)
        
        test_iou = test(model, test_loader, criterion)

        fold_val_ious.append(val_iou)
        fold_test_ious.append(test_iou)

        print(f"Fold {fold} Episode {ep_idx+1}: "
              f"val_IoU={val_iou:.4f}  test_IoU={test_iou:.4f}")

    # ── Final result for Fold 0 only ─────────────────────────────────
    fold_val_mean  = float(np.mean(fold_val_ious))
    fold_test_mean = float(np.mean(fold_test_ious))

    print(f"\n{'='*70}")
    print(f"FINAL RESULTS — FOLD 0 (5 episodes)")
    print(f"{'='*70}")
    print(f"  Fold 0: val={fold_val_mean:.4f}  test={fold_test_mean:.4f}")
    print(f"\nOverall val  mean-IoU: {fold_val_mean:.4f} ± {np.std(fold_val_ious):.4f}")
    print(f"Overall test mean-IoU: {fold_test_mean:.4f} ± {np.std(fold_test_ious):.4f}")
    print(f"\nConfig: Pascal-5i, {K_SHOT}-shot, backbone={BACKBONE}, Fold 0 only")
    print("=" * 70)
