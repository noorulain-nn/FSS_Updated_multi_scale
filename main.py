#!pip install torch, torchdivision, matplotlib, seaborn, numpy, sklearn, pandas

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50, wide_resnet50_2, resnet101, wide_resnet101_2, resnet34, resnet152, vgg16, vgg19
from torchvision.models import vgg16
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score
from torchvision.datasets import CIFAR100, CIFAR10, Omniglot, Caltech101
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from torch.optim.lr_scheduler import StepLR

import Data_Loader
import PLOT
import Models
import APM

import os
import shutil
import random
from collections import defaultdict
#!pip install umap-learn
#import umap
import cv2
from sklearn.preprocessing import label_binarize

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GPU determinism for reproducibility
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Constants
NUM_CLASSES = 5  # Number of classes in ImageNet OR CIFAR-100
FEATURE_DIM = 512  # Placeholder; actual dim auto-detected from backbone (2208 for DenseNet161)
BATCH_SIZE = 128  # Per Table 3 of APM paper
NUM_EPOCHS = 10  # Per Table 3 of APM paper (Epochs per Episode)
LEARNING_RATE = 0.001 # Per Table 3 of APM paper
RANDOM_SEEDS = [42, 142, 242, 342, 442]  # Per Table 3 of APM paper (5 episodes)

# data_path refers to the path of your dataset.
# data_path = path to the data directory
# APM Paper uses CIFAR100 with DenseNet161 backbone
# NOTE: Initial data loading moved inside episode loop for proper multiprocessing
# train_loader, eval_loader, test_loader, NUM_CLASSES = Data_Loader.prepare_cifar(num_classes=5, samples_per_class=5)
# train_loader, eval_loader, test_loader, NUM_CLASSES = Data_Loader.prepare_cub(data_path)
# train_loader, eval_loader, test_loader, NUM_CLASSES = Data_Loader.prepare_caltech(data_path)
# train_loader, eval_loader, test_loader, NUM_CLASSES = Data_Loader.prepare_eurosat(data_path)
# train_loader, eval_loader, test_loader, NUM_CLASSES = Data_Loader.prepare_omniglot()

def validate(model, val_loader, criterion, return_loss=False):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0  # Total number of samples
    all_labels = []
    all_preds = []
    
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass without disabling gradient tracking
        predicted_label, features, attention_scores, predicted_slot = model(images)
        
        # Calculate loss
        loss = criterion(attention_scores, labels)
        val_loss += loss.item() * labels.size(0)
        
        # For calculating accuracy
        predicted_label_tensor = torch.tensor(predicted_label).to(device)
        total += labels.size(0)
        correct += predicted_label_tensor.eq(labels).sum().item()
        
        # Store labels and predictions
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted_label_tensor.cpu().numpy())

        
        
    # Average validation loss
    val_loss /= total
    accuracy = 100. * correct / total
    
    # Calculate precision, recall, and F1 score
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=1)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)
    
    if return_loss:
        return val_loss, accuracy, precision, recall, f1, all_labels, all_preds
    else:
        return accuracy, precision, recall, f1, all_labels, all_preds


# Define the Backbone
# APM Paper uses DenseNet161 per Table 3
backbone_name = "densenet161"

# Initialize criterion (reused across episodes)
criterion = nn.CrossEntropyLoss()

# Storage for episode results (per Table 3: average over 5 episodes)
episode_val_accs = []
episode_test_accs = []
episode_val_prec = []
episode_test_prec = []


#Training Function

# Training Function
# Flush Every Dataset

# Episode specific parameters
episode = 0
ep_precisions = [[] for _ in range(5)]
ep_recalls = [[] for _ in range(5)]
ep_f1_scores = [[] for _ in range(5)]
ep_val_accuracies = [[] for _ in range(5)]
ep_val_losses = [[] for _ in range(5)]
ep_train_accs = [[] for _ in range(5)]
ep_train_losses = [[] for _ in range(5)]
ep_av_val_acc, ep_av_val_pre, ep_av_val_rec, ep_av_val_f1 = 0, 0, 0, 0

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, episode):
    model.train()  # Set model to training mode
    global ep_precisions
    global ep_recalls
    global ep_f1_scores
    global ep_val_accuracies
    global ep_val_losses
    global ep_train_accs
    global ep_train_losses
    global ep_av_val_acc 
    global ep_av_val_pre 
    global ep_av_val_rec 
    global ep_av_val_f1
    
    train_losses = []
    val_losses = []
    topk_val_errors = []
    precisions = []
    recalls = []
    f1_scores = []
    train_accs = []
    val_accs = []
    average = 0
    val_acc = 0
    precision, recall, f1 = 0, 0, 0
    av_pre, av_re, av_f1 = 0, 0, 0
    
   
    for epoch in range(num_epochs):
        
        conflict = 0
        count=0
        running_loss = 0.0  # Track total loss for the epoch
        correct = 0  # Track correct predictions for accuracy
        total = 0  # Track total number of samples
        all_labels = []
        all_outputs = []
        
        # Iterate over the training data loader
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()  # Zero the gradients before backpropagation

            # Forward pass through the model
            predicted_label, features, attention_scores, predicted_slot = model(images)

            # Calculate loss
            memory_loss=0
            
            loss = criterion(attention_scores, labels)
            loss.backward()  # Backpropagate the loss | ORIGINALLY
            optimizer.step()  # Update the model parameters | ORIGINALLY

            # Update running loss
            running_loss += loss.item() * labels.size(0)  # Multiply by batch size to track total loss

            # Calculate training accuracy
            predicted_label_tensor = torch.tensor(predicted_label).to(device)
            total += labels.size(0)  # Total number of samples
            correct += predicted_label_tensor.eq(labels).sum().item()  # Number of correct predictions

            # Memory update
            for i in range(labels.size(0)):
                
                true_label = labels[i].item()
                predicted_slot_i = predicted_slot[i].item()
                predicted_label_i = predicted_label[i]
                attention_i = attention_scores[i]
                features_i = features[i]
                attention_scores_i = attention_scores[i]
                
                if true_label not in model.memory_module.memory_labels and model.memory_module.memory_labels[predicted_slot_i] != -1:
                    mask = torch.tensor([1 if label == -1 else 0 for label in model.memory_module.memory_labels], device=features.device)
                    
                    for i in range(len(mask)):
                        attention_i[i] = attention_i[i] * mask[i]
                    
                    predicted_slot_i = torch.argmax(attention_i).item()
                    conflict += 1
  
                
                memory_loss = model.memory_module.update_memory(features_i, attention_scores_i, true_label, predicted_slot_i)
            
                
        print(f'Total Conflicts: {conflict}')
           
        # Compute epoch loss and accuracy
        epoch_loss = running_loss / total  # Average loss for the epoch
        epoch_accuracy = 100. * correct / total  # Accuracy percentage for the epoch
        
        train_losses.append(epoch_loss)
        train_accs.append(epoch_accuracy)
        val_loss, val_acc, precision, recall, f1, all_labels, all_outputs = validate(model, val_loader, criterion, return_loss=True)#, top1_error, top2_error, top3_error, top4_error, top5_error, top6_error, top7_error, top8_error, top9_error, top10_error = validate(model, val_loader, criterion, return_loss=True)
        
        average += val_acc
        av_pre += precision
        av_re += recall
        av_f1 += f1

        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        current_lr = optimizer.param_groups[0]['lr']
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.2f}%, LR: {current_lr}')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%')
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")
        print(f"\n")
        
        # Step the learning rate scheduler
        scheduler.step()
     
        
    val_precision, val_recall, val_f1, val_accuracy = av_pre/num_epochs, av_re/num_epochs, av_f1/num_epochs, average/num_epochs # Metrics for Histogram
    print(f"Average Training Accuracy: {average/num_epochs}") # Training Metric
    print(f"Average Precision: {val_precision}, Average Recall: {val_recall}, Average F1 Score: {val_f1}")
    
    # Episode specific Training Metrics for later comparison with testing metrics
    ep_precisions[episode].append(precisions)
    ep_recalls[episode].append(recalls)
    ep_f1_scores[episode].append(f1_scores)
    ep_val_accuracies[episode].append(val_accs)
    ep_val_losses[episode].append(val_losses)
    ep_train_accs[episode].append(train_accs)
    ep_train_losses[episode].append(train_losses)
    
    
    ep_av_val_acc += val_accuracy
    ep_av_val_pre += val_precision 
    ep_av_val_rec += val_recall 
    ep_av_val_f1 += val_f1
    
    # Plot Episode Specific Training Metrics
    PLOT.plot_bias_variance_curve(train_losses, val_losses) # Training Metric
    PLOT.plot_metrics_acc(precisions, recalls, f1_scores, val_accs) # Training Metric
    PLOT.plot_metrics(precisions, recalls, f1_scores) # Training Metric
    PLOT.plot_accuracy(train_accs, val_accs) # Training Metric
    #PLOT.plot_class_separation() # Training Metric
    
    #torch.save(model, "model.pth")
    return val_precision, val_recall, val_f1, val_accuracy

# Test Function - Must be defined before episode loop
def test(model, test_loader, criterion):
    model.eval()
    
    test_loss = 0
    correct = 0
    total = 0  # Total number of samples
    all_labels = []
    all_preds = []
    all_probs = []  # To store predicted probabilities

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
    
        # Forward pass through the model
        predicted_label, features, attention_scores, predicted_slot = model(images)
    
        # Apply softmax to get class probabilities
        pred_probs = attention_scores
    
        # Calculate loss using the attention scores and true labels
        loss = criterion(attention_scores, labels)
        test_loss += loss.item() * labels.size(0)  # Accumulate loss
    
        # Get the predicted class
        predicted_label_tensor = torch.tensor(predicted_label).to(device)
    
        # Update total and correct counts for accuracy
        total += labels.size(0)
        correct += predicted_label_tensor.eq(labels).sum().item()
    
        # Store labels, predictions, and probabilities for metrics calculation
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted_label_tensor.cpu().numpy())
        all_probs.extend(pred_probs.detach().cpu().numpy())  # Save probabilities

    # Calculate average test loss
    test_loss /= total
    accuracy = 100. * correct / total

    # Calculate precision, recall, and F1 score (weighted average)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=1)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
    print(f'Test Precision: {precision:.4f}, Test Recall: {recall:.4f}, Test F1 Score: {f1:.4f}')
    
    # Return test metrics for aggregation across episodes
    return accuracy, precision, recall, f1

# ============================================================================
# EPISODE LOOP - Per Table 3: "All results are averaged over 5 episodes"
# Per Experimental Setup: Reload pretrained backbone & re-initialize model
# at start of each episode, with random seeds 42, 142, 242, 342, 442
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("STARTING APM REPRODUCTION: 5 EPISODES WITH RANDOM SEEDS")
    print("="*80 + "\n")

    for episode_idx, seed in enumerate(RANDOM_SEEDS):
        print(f"\n{'='*80}")
        print(f"EPISODE {episode_idx + 1}/{len(RANDOM_SEEDS)} | Random Seed: {seed}")
        print(f"{'='*80}\n")
        
        # Set random seed for reproducibility (per paper section "Experimental setup")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # CRITICAL: Reload data with current seed
        # This ensures different class splits/samples for each episode
        print(f"Loading CIFAR100 dataset with seed {seed}...")
        train_loader, eval_loader, test_loader, NUM_CLASSES = Data_Loader.prepare_cifar(num_classes=5, samples_per_class=1, seed=seed)
        
        # CRITICAL: Reload pretrained backbone at start of each episode
        # Per paper: "reload the pretrained backbone at the start of each episode"
        print(f"Loading fresh DenseNet161 backbone with ImageNet pretrained weights...")
        backbone, FEATURE_DIM = Models.load_backbone(backbone_name)
        
        # CRITICAL: Re-initialize model at start of each episode
        # Per paper: "randomly re-initialize the models"
        print(f"Re-initializing APM model...")
        model = APM.MemoryEnabledCNN(backbone, NUM_CLASSES, FEATURE_DIM)
        model = model.to(device)
        
        # Re-initialize optimizer and scheduler for fresh episode
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.30)
        
        # Reset episode counters
        episode = episode_idx
        
        # Training
        print(f"\nTraining Episode {episode + 1}...")
        val_precision, val_recall, val_f1, val_accuracy = train(model, train_loader, eval_loader, criterion, optimizer, scheduler, 10, episode)
        
        # Store validation results
        episode_val_accs.append(val_accuracy)
        episode_val_prec.append(val_precision)
        
        # Test
        print(f"\nTesting Episode {episode + 1}...")
        test_accuracy, test_precision, test_recall, test_f1 = test(model, test_loader, criterion)
        
        # Store test results
        episode_test_accs.append(test_accuracy)
        episode_test_prec.append(test_precision)
        
        print(f"\nEpisode {episode + 1} Complete!")
        print(f"Val Accuracy: {val_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")
        print(f"Val Precision: {val_precision:.4f}, Test Precision: {test_precision:.4f}\n")

    # ============================================================================
    # FINAL RESULTS - Averaged across all 5 episodes
    # ============================================================================
    print("\n" + "="*80)
    print("FINAL RESULTS - AVERAGED ACROSS 5 EPISODES (Per Table 3)")
    print("="*80 + "\n")

    mean_val_acc = np.mean(episode_val_accs)
    std_val_acc = np.std(episode_val_accs)
    mean_val_prec = np.mean(episode_val_prec)
    std_val_prec = np.std(episode_val_prec)
    
    mean_test_acc = np.mean(episode_test_accs)
    std_test_acc = np.std(episode_test_accs)
    mean_test_prec = np.mean(episode_test_prec)
    std_test_prec = np.std(episode_test_prec)

    print("VALIDATION RESULTS:")
    print(f"  Mean Accuracy: {mean_val_acc:.2f}% (±{std_val_acc:.2f}%)")
    print(f"  Mean Precision: {mean_val_prec:.4f} (±{std_val_prec:.4f})")
    
    print("\nTEST RESULTS:")
    print(f"  Mean Accuracy: {mean_test_acc:.2f}% (±{std_test_acc:.2f}%)")
    print(f"  Mean Precision: {mean_test_prec:.4f} (±{std_test_prec:.4f})")
    
    print(f"\nIndividual Episode Val Accuracies: {[f'{acc:.2f}%' for acc in episode_val_accs]}")
    print(f"Individual Episode Test Accuracies: {[f'{acc:.2f}%' for acc in episode_test_accs]}")
    print(f"\nConfig: CIFAR100, DenseNet161, 5-way 1-shot, Batch Size 128, 10 epochs/episode")
    print("="*80 + "\n")

    