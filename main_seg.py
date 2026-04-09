"""
APM Few-Shot Segmentation — main_seg.py

Methodology matches APM classification exactly:
  - Support images used for BOTH loss computation AND memory update
  - Query images used for evaluation only (no gradient)
  - Same hyperparameters as APM paper Table 3
  - Same 5-episode structure with seeds 42/142/242/342/442
  - Only last backbone block fine-tuned
"""

"""
APM Few-Shot Segmentation — main_seg.py
Updated with proper channel projection handling for multi-scale decoder.
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
    torch.backends.cudnn.benchmark = False

# ── Config — same as APM paper Table 3 ─────────────────────────
DATA_ROOT = "./data/fss-data"
FOLD = 0
K_SHOT = 1
IMG_SIZE = 473
NUM_CLASSES = 1
BATCH_SIZE = 6
NUM_EPOCHS = 10
LR = 0.0005
RANDOM_SEEDS = [42, 142, 242, 342, 442]
BACKBONE = "resnet50"

# ── Metric: mean-IoU ────────────────────────────────────────────
def compute_iou(pred_mask: torch.Tensor, true_mask: torch.Tensor) -> float:
    pred = pred_mask.bool()
    true = true_mask.bool()
    inter = (pred & true).sum().float()
    union = (pred | true).sum().float()
    return (inter / (union + 1e-6)).item()

# ── Validation function ─────────────────────────────────────────
def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    count = 0
    
    with torch.no_grad():
        for s_imgs, s_masks, q_imgs, q_masks in val_loader:
            B = s_imgs.shape[0]
            
            # Reset memory for this episode
            model.memory_module.initialized = [False] * NUM_CLASSES
            
            # Populate memory from support images
            for b in range(B):
                img = s_imgs[b, 0].unsqueeze(0).to(device)  # [1, 3, H, W]
                mask = s_masks[b, 0]                        # [H, W]
                
                # encode returns: reduced4 (256ch for decoder), reduced3 (256ch), raw4 (2048ch for memory)
                feats4_reduced, feats3_reduced, feats4_raw = model.encode(img)
                
                # Use RAW high-dim features for better prototype quality
                model.memory_module.update_memory(
                    feats4_raw, 
                    mask, 
                    class_label=0
                )
            
            # Evaluate on query images
            q_imgs = q_imgs.to(device)
            q_masks = q_masks.long().to(device)
            
            seg_logits, _, _ = model(q_imgs)
            
            loss = criterion(seg_logits, q_masks)
            total_loss += loss.item() * B
            
            pred = seg_logits.argmax(dim=1)
            for b in range(B):
                total_iou += compute_iou(pred[b].cpu(), q_masks[b].cpu())
            
            count += B
            
    return total_loss / count, total_iou / count

# ── Training function ───────────────────────────────────────────
def train(model, train_loader, val_loader,
          criterion, optimizer, scheduler,
          num_epochs, episode):
    train_losses, val_losses = [], []
    train_ious, val_ious = [], []
    
    for epoch in range(num_epochs):
        model.train()
        run_loss = 0.0
        run_iou = 0.0
        count = 0
        
        for s_imgs, s_masks, q_imgs, q_masks in train_loader:
            B = s_imgs.shape[0]
            support_img = s_imgs[:, 0].to(device)      # [B, 3, H, W]
            support_mask = s_masks[:, 0]               # [B, H, W]
            
            optimizer.zero_grad()
            
            # Forward pass
            seg_logits, features, _ = model(support_img)   # features = feats4_raw (2048ch)
            
            # Loss on support
            loss = criterion(seg_logits, support_mask.long().to(device))
            loss.backward()
            optimizer.step()
            
            # Memory update (no gradient)
            with torch.no_grad():
                for b in range(B):
                    model.memory_module.update_memory(
                        features[b:b+1].detach(),   # [1, 2048, H', W']
                        support_mask[b],
                        class_label=0
                    )
            
            # Metrics
            pred = seg_logits.argmax(dim=1)
            run_loss += loss.item() * B
            for b in range(B):
                run_iou += compute_iou(pred[b].cpu(), support_mask[b].cpu())
            count += B
        
        # Epoch summary
        epoch_loss = run_loss / count
        epoch_iou = run_iou / count
        val_loss, val_iou = validate(model, val_loader, criterion)
        
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        train_ious.append(epoch_iou)
        val_ious.append(val_iou)
        
        print(f"Episode {episode+1} | Epoch [{epoch+1}/{num_epochs}] "
              f"train_loss={epoch_loss:.4f} train_IoU={epoch_iou:.4f} "
              f"val_loss={val_loss:.4f} val_IoU={val_iou:.4f}")
        
        scheduler.step()
    
    # Plots
    PLOT.plot_bias_variance_curve(train_losses, val_losses)
    PLOT.plot_accuracy(train_ious, val_ious)
    return float(np.mean(val_ious))

# ── Test function ───────────────────────────────────────────────
def test(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    count = 0
    
    with torch.no_grad():
        for s_imgs, s_masks, q_imgs, q_masks in test_loader:
            B = s_imgs.shape[0]
            
            # Reset memory
            model.memory_module.initialized = [False] * NUM_CLASSES
            
            # Populate memory from support
            for b in range(B):
                img = s_imgs[b, 0].unsqueeze(0).to(device)
                mask = s_masks[b, 0]
                
                feats4_reduced, feats3_reduced, feats4_raw = model.encode(img)
                
                model.memory_module.update_memory(
                    feats4_raw, 
                    mask, 
                    class_label=0
                )
            
            # Evaluate on query
            q_imgs = q_imgs.to(device)
            q_masks = q_masks.long().to(device)
            
            seg_logits, _, _ = model(q_imgs)
            
            loss = criterion(seg_logits, q_masks)
            total_loss += loss.item() * B
            
            pred = seg_logits.argmax(dim=1)
            for b in range(B):
                total_iou += compute_iou(pred[b].cpu(), q_masks[b].cpu())
            
            count += B
    
    test_loss = total_loss / count
    test_iou = total_iou / count
    print(f"Test loss={test_loss:.4f} mean-IoU={test_iou:.4f}")
    return test_iou

# ── Episode loop (Fold 0 only) ──────────────────────────────────
if __name__ == '__main__':
    print("=" * 70)
    print("APM Few-Shot Segmentation — FOLD 0 ONLY × 5 episodes")
    print("=" * 70)
    
    # Use DiceLoss for binary segmentation
    class DiceLoss(nn.Module):
        def forward(self, pred, target):
            pred = torch.sigmoid(pred[:, 1])  # foreground channel
            target = (target == 1).float()
            intersection = (pred * target).sum()
            return 1 - (2 * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
    
    criterion = DiceLoss()
    fold = 0
    print(f"\n{'#'*70}")
    print(f"RUNNING ONLY FOLD {fold}")
    print(f"{'#'*70}")
    
    fold_val_ious = []
    fold_test_ious = []
    
    for ep_idx, seed in enumerate(RANDOM_SEEDS):
        print(f"\n{'='*70}")
        print(f"FOLD {fold} | EPISODE {ep_idx+1}/5 | seed={seed}")
        print(f"{'='*70}\n")
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Load data
        train_loader, val_loader, test_loader, _ = Data_Loader.prepare_pascal5i(
            DATA_ROOT,
            fold=fold,
            k_shot=K_SHOT,
            img_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            seed=seed
        )
        
        # Create model
        backbone, feat_dim = Models.load_backbone_seg(BACKBONE)
        model = APM.SegAPM(
            backbone,
            num_classes=NUM_CLASSES,
            feature_dim=feat_dim,
            output_size=(IMG_SIZE, IMG_SIZE)
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.30)
        
        val_iou = train(model, train_loader, val_loader,
                        criterion, optimizer, scheduler,
                        NUM_EPOCHS, ep_idx)
        
        test_iou = test(model, test_loader, criterion)
        
        fold_val_ious.append(val_iou)
        fold_test_ious.append(test_iou)
        
        print(f"Fold {fold} Episode {ep_idx+1}: "
              f"val_IoU={val_iou:.4f} test_IoU={test_iou:.4f}")
    
    # Final result for Fold 0
    fold_val_mean = float(np.mean(fold_val_ious))
    fold_test_mean = float(np.mean(fold_test_ious))
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS — FOLD 0 (5 episodes)")
    print(f"{'='*70}")
    print(f" Fold 0: val={fold_val_mean:.4f} test={fold_test_mean:.4f}")
    print(f"\nOverall val mean-IoU: {fold_val_mean:.4f} ± {np.std(fold_val_ious):.4f}")
    print(f"Overall test mean-IoU: {fold_test_mean:.4f} ± {np.std(fold_test_ious):.4f}")
    print(f"\nConfig: Pascal-5i, {K_SHOT}-shot, backbone={BACKBONE}, Fold 0 only")
    print("=" * 70)
