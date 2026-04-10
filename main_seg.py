"""
APM Few-Shot Segmentation — main_seg.py  (all 4 folds x 5 episodes)

FIXES APPLIED:
  Issue 4 — initialized reset is now per-episode, not per-batch.
  Issue 5 — val_IoU reported as best epoch (max), not mean over epochs.
  Issue 7 — both background (class_label=0) AND foreground (class_label=1)
             prototypes updated in train, validate, and test.

RUN CONFIG:
  4 folds x 5 episodes x 20 epochs = 400 total epochs
  1-way 5-shot binary segmentation on Pascal-5i
"""

import torch
import torch.nn as nn
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ── Config ──────────────────────────────────────────────────────
DATA_ROOT    = "./data/fss-data"
K_SHOT       = 5
IMG_SIZE     = 473
NUM_CLASSES  = 1          # passed to SegAPM; internally forced to 2 in APM.py
BATCH_SIZE   = 6
NUM_EPOCHS   = 20
LR           = 0.0005
RANDOM_SEEDS = [42, 142, 242, 342, 442]
BACKBONE     = "resnet50"


# ── IoU Metric ──────────────────────────────────────────────────
def compute_iou(pred_mask: torch.Tensor, true_mask: torch.Tensor) -> float:
    pred  = pred_mask.bool()
    true  = true_mask.bool()
    inter = (pred & true).sum().float()
    union = (pred | true).sum().float()
    return (inter / (union + 1e-6)).item()


# ── Combined Loss (CE + Dice) ───────────────────────────────────
class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.4):
        super().__init__()
        self.ce          = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        ce_loss   = self.ce(pred, target.long())
        pred_fg   = torch.sigmoid(pred[:, 1])
        target_fg = (target == 1).float()
        inter     = (pred_fg * target_fg).sum()
        dice_loss = 1 - (2 * inter + 1e-6) / (pred_fg.sum() + target_fg.sum() + 1e-6)
        return ce_loss + self.dice_weight * dice_loss


# ── Validate ────────────────────────────────────────────────────
def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    total_iou  = 0.0
    count      = 0

    # Issue 4 fix: reset ONCE per validation call, not per batch
    model.memory_module.initialized = [False] * model.memory_module.num_classes

    with torch.no_grad():
        for s_imgs, s_masks, q_imgs, q_masks in val_loader:
            B = s_imgs.shape[0]

            for b in range(B):
                for shot in range(K_SHOT):
                    img  = s_imgs[b, shot].unsqueeze(0).to(device)
                    mask = s_masks[b, shot]                    # (H, W) CPU
                    _, _, feats4_raw = model.encode(img)

                    # Issue 7 fix: update BOTH fg and bg prototypes
                    fg_mask = (mask == 1)
                    bg_mask = (mask == 0)
                    if fg_mask.any():
                        model.memory_module.update_memory(
                            feats4_raw, fg_mask.long(), class_label=1
                        )
                    if bg_mask.any():
                        model.memory_module.update_memory(
                            feats4_raw, bg_mask.long(), class_label=0
                        )

            q_imgs  = q_imgs.to(device)
            q_masks = q_masks.long().to(device)

            seg_logits, _, _ = model(q_imgs)
            loss = criterion(seg_logits, q_masks)
            total_loss += loss.item() * B

            pred = seg_logits.argmax(dim=1)
            for b in range(B):
                total_iou += compute_iou(pred[b].cpu(), q_masks[b].cpu())
            count += B

    return total_loss / count, total_iou / count


# ── Training ────────────────────────────────────────────────────
def train(model, train_loader, val_loader,
          criterion, optimizer, scheduler,
          num_epochs, episode, fold):
    train_losses, val_losses = [], []
    train_ious,   val_ious   = [], []

    for epoch in range(num_epochs):
        model.train()
        run_loss = 0.0
        run_iou  = 0.0
        count    = 0

        for s_imgs, s_masks, q_imgs, q_masks in train_loader:
            B = s_imgs.shape[0]

            support_img  = s_imgs[:, 0].to(device)        # (B, 3, H, W)
            support_mask = s_masks[:, 0].to(device)        # (B, H, W)
            query_img    = q_imgs.to(device)               # (B, 3, H, W)
            query_mask   = q_masks.long().to(device)       # (B, H, W)

            optimizer.zero_grad()

            # Forward on support
            seg_logits_support, features_support, _ = model(support_img)
            loss_support = criterion(seg_logits_support, support_mask)

            # Forward on query
            seg_logits_query, _, _ = model(query_img)
            loss_query = criterion(seg_logits_query, query_mask)

            # Combined loss: 70% support + 30% query
            loss = 0.7 * loss_support + 0.3 * loss_query
            loss.backward()
            optimizer.step()

            # Memory update AFTER optimizer.step (matches APM paper order)
            with torch.no_grad():
                sf = features_support.detach()
                for b in range(B):
                    feat_b = sf[b:b+1]

                    # Issue 7 fix: update BOTH fg and bg prototypes
                    fg_mask = (support_mask[b] == 1)
                    bg_mask = (support_mask[b] == 0)
                    if fg_mask.any():
                        model.memory_module.update_memory(
                            feat_b, fg_mask.long().cpu(), class_label=1
                        )
                    if bg_mask.any():
                        model.memory_module.update_memory(
                            feat_b, bg_mask.long().cpu(), class_label=0
                        )

            pred     = seg_logits_support.argmax(dim=1)
            run_loss += loss.item() * B
            for b in range(B):
                run_iou += compute_iou(pred[b].cpu(), support_mask[b].cpu())
            count += B

        epoch_loss = run_loss / count
        epoch_iou  = run_iou  / count
        val_loss, val_iou = validate(model, val_loader, criterion)

        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        train_ious.append(epoch_iou)
        val_ious.append(val_iou)

        print(f"Fold {fold} | Episode {episode+1} | Epoch [{epoch+1}/{num_epochs}] "
              f"train_loss={epoch_loss:.4f} train_IoU={epoch_iou:.4f} "
              f"val_loss={val_loss:.4f} val_IoU={val_iou:.4f}")

        scheduler.step()

    PLOT.plot_bias_variance_curve(train_losses, val_losses)
    PLOT.plot_accuracy(train_ious, val_ious)

    # Issue 5 fix: return BEST val epoch, not mean
    return float(max(val_ious))


# ── Test ────────────────────────────────────────────────────────
def test(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    total_iou  = 0.0
    count      = 0

    # Issue 4 fix: reset ONCE per test call, not per batch
    model.memory_module.initialized = [False] * model.memory_module.num_classes

    with torch.no_grad():
        for s_imgs, s_masks, q_imgs, q_masks in test_loader:
            B = s_imgs.shape[0]

            for b in range(B):
                for shot in range(K_SHOT):
                    img  = s_imgs[b, shot].unsqueeze(0).to(device)
                    mask = s_masks[b, shot]                    # (H, W) CPU
                    _, _, feats4_raw = model.encode(img)

                    # Issue 7 fix: update BOTH fg and bg prototypes
                    fg_mask = (mask == 1)
                    bg_mask = (mask == 0)
                    if fg_mask.any():
                        model.memory_module.update_memory(
                            feats4_raw, fg_mask.long(), class_label=1
                        )
                    if bg_mask.any():
                        model.memory_module.update_memory(
                            feats4_raw, bg_mask.long(), class_label=0
                        )

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
    print(f"Test loss={test_loss:.4f} mean-IoU={test_iou:.4f}")
    return test_iou


# ── Main Loop — ALL 4 FOLDS x 5 EPISODES ───────────────────────
if __name__ == '__main__':
    print("=" * 70)
    print("APM Few-Shot Segmentation — ALL 4 FOLDS x 5 EPISODES")
    print(f"Config: {K_SHOT}-shot, backbone={BACKBONE}, epochs={NUM_EPOCHS}")
    print("=" * 70)

    criterion     = CombinedLoss(dice_weight=0.4)
    all_fold_val  = []
    all_fold_test = []

    for fold in range(4):                          # fold 0, 1, 2, 3
        print(f"\n{'#'*70}")
        print(f"FOLD {fold} / 3")
        print(f"{'#'*70}")

        fold_val_ious  = []
        fold_test_ious = []

        for ep_idx, seed in enumerate(RANDOM_SEEDS):    # 5 episodes per fold
            print(f"\n{'='*70}")
            print(f"FOLD {fold} | EPISODE {ep_idx+1}/5 | seed={seed}")
            print(f"{'='*70}\n")

            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            train_loader, val_loader, test_loader, _ = Data_Loader.prepare_pascal5i(
                DATA_ROOT, fold=fold, k_shot=K_SHOT, img_size=IMG_SIZE,
                batch_size=BATCH_SIZE, seed=seed
            )

            backbone, feat_dim = Models.load_backbone_seg(BACKBONE)
            model = APM.SegAPM(
                backbone, num_classes=NUM_CLASSES, feature_dim=feat_dim,
                output_size=(IMG_SIZE, IMG_SIZE)
            ).to(device)

            optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
            scheduler = StepLR(optimizer, step_size=8, gamma=0.8)

            val_iou  = train(model, train_loader, val_loader,
                             criterion, optimizer, scheduler,
                             NUM_EPOCHS, ep_idx, fold)
            test_iou = test(model, test_loader, criterion)

            fold_val_ious.append(val_iou)
            fold_test_ious.append(test_iou)

            print(f"Fold {fold} Episode {ep_idx+1}: "
                  f"val_IoU={val_iou:.4f} (best epoch)  test_IoU={test_iou:.4f}")

        # ── Per-fold summary ─────────────────────────────────────
        fold_val_mean  = float(np.mean(fold_val_ious))
        fold_test_mean = float(np.mean(fold_test_ious))
        all_fold_val.append(fold_val_mean)
        all_fold_test.append(fold_test_mean)

        print(f"\nFold {fold} summary (5 episodes):")
        print(f"  val  mean-IoU : {fold_val_mean:.4f}  +- {np.std(fold_val_ious):.4f}")
        print(f"  test mean-IoU : {fold_test_mean:.4f}  +- {np.std(fold_test_ious):.4f}")

    # ── Final summary across all 4 folds ─────────────────────────
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS — ALL 4 FOLDS (4 x 5 = 20 episodes)")
    print(f"{'='*70}")
    for f_idx, (v, t) in enumerate(zip(all_fold_val, all_fold_test)):
        print(f"  Fold {f_idx}: val={v:.4f}  test={t:.4f}")
    print(f"\nOverall val  mean-IoU : {np.mean(all_fold_val):.4f}  "
          f"+- {np.std(all_fold_val):.4f}")
    print(f"Overall test mean-IoU : {np.mean(all_fold_test):.4f}  "
          f"+- {np.std(all_fold_test):.4f}")
    print(f"\nConfig: Pascal-5i all folds, {K_SHOT}-shot, backbone={BACKBONE}")
    print("=" * 70)
