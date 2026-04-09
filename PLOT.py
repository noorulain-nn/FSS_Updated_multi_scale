import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from matplotlib.colors import LogNorm

# Function to plot Bias-Variance curve
def plot_bias_variance_curve(train_losses, val_losses):
    plt.figure(figsize=(10, 8), dpi=300)
    plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
    plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.title(f'Bias-Variance Curve', fontsize=25)
    plt.legend(fontsize=12)
    name=f"Bias_Variance_Curve.png"
    plt.tight_layout()
    plt.savefig(name, dpi=300)
    #plt.show()
    plt.close()

# Function to plot accuracies over epochs
def plot_accuracy(train_accuracies, val_accuracies):
    plt.figure(figsize=(10, 8), dpi=300)
    plt.plot(range(len(train_accuracies)), train_accuracies, label='Train Accuracy')
    plt.plot(range(len(val_accuracies)), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Accuracy (%)', fontsize=15)
    plt.title('Training and Validation Accuracy', fontsize=25)
    plt.legend(fontsize=12)
    name=f"Training_and_Validation_Accuracy.png"
    plt.tight_layout()
    plt.savefig(name, dpi=300)
    #plt.show()
    plt.close()


# Function to plot confusion matrix
def plot_confusion_matrix(true_labels, pred_labels, classes):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, norm=LogNorm())
    plt.xlabel('Predicted Labels', fontsize=15)
    plt.ylabel('True Labels', fontsize=15)
    plt.title('Confusion Matrix', fontsize=25)
    name=f"Confusion_Matrix.png"
    plt.tight_layout()
    plt.savefig(name, dpi=300)
    #plt.show()
    plt.close()

# ROC Curve
def plot_roc_curve(true_labels, pred_probs, num_classes):
    true_labels_onehot = label_binarize(true_labels, classes=list(range(num_classes)))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Compute ROC for each class
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(true_labels_onehot[:, i], pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(true_labels_onehot.ravel(), pred_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curves
    plt.figure(figsize=(10, 8), dpi=300)
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    plt.plot(fpr["macro"], tpr["macro"], label=f'Macro-Average (AUC = {roc_auc["macro"]:.2f})', linestyle='--', linewidth=2)
    plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-Average (AUC = {roc_auc["micro"]:.2f})', linestyle=':', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.title('ROC Curve', fontsize=20)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig("ROC_Curve.png", dpi=300)
    plt.close()

# Function to plot Precision-Recall curve
def plot_precision_recall_curve(true_labels, pred_probs, num_classes):
    true_labels_onehot = label_binarize(true_labels, classes=list(range(num_classes)))
    precision = dict()
    recall = dict()
    average_precision = dict()

    # Compute Precision-Recall for each class
    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(true_labels_onehot[:, i], pred_probs[:, i])
        average_precision[i] = average_precision_score(true_labels_onehot[:, i], pred_probs[:, i])

    # Compute macro-average Precision-Recall curve
    precision["macro"], recall["macro"], _ = precision_recall_curve(
        true_labels_onehot.ravel(), pred_probs.ravel()
    )
    average_precision["macro"] = average_precision_score(true_labels_onehot, pred_probs, average="macro")

    # Compute micro-average Precision-Recall curve
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        true_labels_onehot.ravel(), pred_probs.ravel()
    )
    average_precision["micro"] = average_precision_score(true_labels_onehot, pred_probs, average="micro")

    # Plot Precision-Recall curves
    plt.figure(figsize=(10, 8), dpi=300)
    for i in range(num_classes):
        plt.plot(recall[i], precision[i], label=f'Class {i} (AP = {average_precision[i]:.2f})')
    plt.plot(recall["macro"], precision["macro"], label=f'Macro-Average (AP = {average_precision["macro"]:.2f})', linestyle='--', linewidth=2)
    plt.plot(recall["micro"], precision["micro"], label=f'Micro-Average (AP = {average_precision["micro"]:.2f})', linestyle=':', linewidth=2)
    plt.xlabel('Recall', fontsize=15)
    plt.ylabel('Precision', fontsize=15)
    plt.title('Precision-Recall Curve', fontsize=20)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig("Precision_Recall_Curve.png", dpi=300)
    plt.close()


#Plot for Precision, recall and F1-score
def plot_metrics(precisions, recalls, f1_scores):
    epochs = range(1, len(precisions) + 1)
    plt.figure(figsize=(12, 6), dpi=600)
    plt.plot(epochs, precisions, marker='o', label='Precision')
    plt.plot(epochs, recalls, marker='s', label='Recall')
    plt.plot(epochs, f1_scores, marker='^', label='F1 Score')
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Score', fontsize=15)
    plt.title('Precision, Recall, and F1 Score Over Training Epochs', fontsize=15)
    plt.legend(fontsize=12)
    name=f"Training_Precision_Recall_F1.png"
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(name, dpi=600)
    #plt.show()
    plt.close()


#Plot for Precision, recall, f1 and accuracy
def plot_metrics_acc(precisions, recalls, f1_scores, accuracies):
    epochs = range(1, len(precisions) + 1)
    if not (len(precisions) == len(recalls) == len(f1_scores) == len(accuracies)):
        raise ValueError("Input lists must have the same length.")
    accuracies = [x / 100 for x in accuracies]
    plt.figure(figsize=(12, 6), dpi=600)
    plt.plot(epochs, precisions, marker='o', label='Precision')
    plt.plot(epochs, recalls, marker='s', label='Recall')
    plt.plot(epochs, f1_scores, marker='^', label='F1 Score')
    plt.plot(epochs, accuracies, marker='d', label='Accuracy')
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Score', fontsize=15)
    plt.title('Precision, Recall, F1 Score, and Accuracy Over Training Epochs', fontsize=15)
    plt.legend(fontsize=10)
    name = f"Training_Precision_Recall_F1_Accuracy.png"
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(name, dpi=600)
    #plt.show()
    plt.close()


# Plot for class boundries
def plot_class_separation():
    # Extract memory vectors from the model
    memory_vectors = model.memory_module.memory.data.cpu().numpy()  # Shape: [num_classes, feature_dim]
    class_labels = np.arange(memory_vectors.shape[0])  # Classes labeled from 0 to num_classes - 1
    
    # Reduce to 2D using UMAP
    reducer = umap.UMAP(n_components=2, random_state=42)
    memory_vectors_2d = reducer.fit_transform(memory_vectors)
    
    # Create a color map
    cmap = plt.get_cmap('tab10')  # 'tab10' has 10 distinct colors
    
    plt.figure(figsize=(10, 8), dpi=300)
    for class_label in np.unique(class_labels):
        indices = class_labels == class_label
        plt.scatter(memory_vectors_2d[indices, 0], memory_vectors_2d[indices, 1],
                    color=cmap(class_label % 10),
                    label=f'Class {class_label}')
    
    plt.legend(title='Classes', fontsize=12)
    plt.title('Visualization of Memory Slots (Class Separation)', fontsize=16)
    plt.xlabel('Dimension 1', fontsize=14)
    plt.ylabel('Dimension 2', fontsize=14)
    plt.grid(True)
    name = 'Class_Separation.png'
    plt.savefig(name, dpi=300)
    #plt.show()
    plt.close()

# Accuracy histogram (for traing and testing comparison)
def plot_accuracy_histogram(train_accuracy, test_accuracy):
    """
    Plots a histogram for validation and test accuracies.
    Args:
        train_accuracy: Validation accuracy.
        test_accuracy: Test accuracy.
    """
    metrics = ['Validation Accuracy', 'Test Accuracy']
    values = [train_accuracy, test_accuracy]

    plt.figure(figsize=(10, 8), dpi=600)
    plt.bar(metrics, values, color=['green', 'orange'])
    plt.ylim(0, 100)
    plt.ylabel('Accuracy (%)', fontsize=15)
    plt.title('Accuracy Histogram', fontsize=18)
    for i, v in enumerate(values):
        plt.text(i, v + 1, f'{v:.2f}%', ha='center', fontsize=12)
    plt.tight_layout()
    plt.savefig("Accuracy_Histogram.png", dpi=600)
    plt.close()

# Precision, Recall, F1 histogram (for traing and testing comparison)
def plot_metrics_histogram(val_precision, val_recall, val_f1, test_precision, test_recall, test_f1):
    """
    Plots a histogram for precision, recall, and F1 score for validation and testing.
    Args:
        val_precision: Validation precision score.
        val_recall: Validation recall score.
        val_f1: Validation F1 score.
        test_precision: Test precision score.
        test_recall: Test recall score.
        test_f1: Test F1 score.
    """
    metrics = ['Precision', 'Recall', 'F1 Score']
    val_values = [val_precision, val_recall, val_f1]
    test_values = [test_precision, test_recall, test_f1]

    x = np.arange(len(metrics))  # the label locations
    width = 0.35  # the width of the bars

    plt.figure(figsize=(10, 8), dpi=600)
    plt.bar(x - width / 2, val_values, width, label='Validation', color='green', alpha=0.7)
    plt.bar(x + width / 2, test_values, width, label='Testing', color='orange', alpha=0.7)

    # Add text for values on top of the bars
    for i in range(len(metrics)):
        plt.text(i - width / 2, val_values[i] + 0.02, f'{val_values[i]:.2f}', ha='center', fontsize=10)
        plt.text(i + width / 2, test_values[i] + 0.02, f'{test_values[i]:.2f}', ha='center', fontsize=10)

    plt.ylim(0, 1.1)
    plt.xticks(x, metrics, fontsize=12)
    plt.ylabel('Score', fontsize=15)
    plt.title('Accuracy, Precision, Recall, and F1 Score Histogram', fontsize=18)
    plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig("Metrics_Histogram.png", dpi=600)
    plt.close()

# Metrics Histogram including Accuracy
def plot_metrics_histogram_acc(val_acc, val_precision, val_recall, val_f1, test_acc, test_precision, test_recall, test_f1):
    """
    Plots a histogram for accuracy, precision, recall, and F1 score for validation and testing.
    Args:
        val_acc: Validation accuracy score.
        val_precision: Validation precision score.
        val_recall: Validation recall score.
        val_f1: Validation F1 score.
        test_acc: Test accuracy score.
        test_precision: Test precision score.
        test_recall: Test recall score.
        test_f1: Test F1 score.
    """
    val_acc /= 100
    test_acc /= 100 
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    val_values = [val_acc, val_precision, val_recall, val_f1]
    test_values = [test_acc, test_precision, test_recall, test_f1]

    x = np.arange(len(metrics))  # the label locations
    width = 0.35  # the width of the bars

    plt.figure(figsize=(12, 8), dpi=300)
    plt.bar(x - width / 2, val_values, width, label='Validation', color='green', alpha=0.7)
    plt.bar(x + width / 2, test_values, width, label='Testing', color='orange', alpha=0.7)

    # Add text for values on top of the bars
    for i in range(len(metrics)):
        plt.text(i - width / 2, val_values[i] + 0.02, f'{val_values[i]:.2f}', ha='center', fontsize=10)
        plt.text(i + width / 2, test_values[i] + 0.02, f'{test_values[i]:.2f}', ha='center', fontsize=10)

    plt.ylim(0, 1.1)
    plt.xticks(x, metrics, fontsize=12)
    plt.ylabel('Score', fontsize=15)
    plt.title('Accuracy, Precision, Recall, and F1 Score Histogram', fontsize=18)
    plt.legend(loc='upper right', fontsize=5)
    plt.tight_layout()
    plt.savefig("Metrics_Histogram_Acc.png", dpi=600)
    #plt.show()
    plt.close()

#Clustered Bar Chart for combined datasets
def plot_metrics_stacked(datasets,
                         val_accuracy, val_precision, val_recall, val_f1,
                         test_accuracy, test_precision, test_recall, test_f1):
    """
    For each dataset and metric (Accuracy, Precision, Recall, F1), plots a single stacked bar:
    - The bottom part (darker color) is the validation score.
    - The top part (lighter color) is the difference (test - validation).

    Assumes test scores are greater than validation scores.
    """
    val_accuracy = [x / 100 for x in val_accuracy]
    test_accuracy = [x / 100 for x in test_accuracy]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    # Arrange the scores in arrays [metric, dataset]
    val_scores = np.array([val_accuracy, val_precision, val_recall, val_f1])
    test_scores = np.array([test_accuracy, test_precision, test_recall, test_f1])
    
    # Calculate the difference
    diff_scores = test_scores - val_scores

    # Define a nice color scheme for each metric
    # Each tuple: (val_color, diff_color)
    colors = {
        'Accuracy':  ('dodgerblue', 'blue'),  # Blue shades
        'Precision': ('limegreen', '#2ca02c'),  # Green shades
        'Recall':    ('gold', '#ff7f0e'),  # Orange/Gold shades
        'F1':        ('orchid', 'purple')  # Purple shades
    }

    n_datasets = len(datasets)
    n_metrics = len(metrics)

    x = np.arange(n_datasets)  # one cluster per dataset
    total_cluster_width = 0.8
    bar_width = total_cluster_width / n_metrics

    plt.figure(figsize=(12, 8), dpi=600)

    # Plot each metric
    for m, metric in enumerate(metrics):
        val_m = val_scores[m, :]
        diff_m = diff_scores[m, :]
        metric_colors = colors[metric]
        offset = -total_cluster_width/2 + m * bar_width + bar_width/2

        # Bottom bar (validation)
        bottom_bars = plt.bar(x + offset, val_m, width=bar_width, color=metric_colors[0], alpha=0.9, label=metric if m == 0 else "")
        # Top bar (difference)
        top_bars = plt.bar(x + offset, diff_m, width=bar_width, bottom=val_m, color=metric_colors[1], alpha=0.9)

        # Add text for validation score inside the bar
        for i in range(n_datasets):
            # Validation label
            plt.text(x[i] + offset, val_m[i] - 0.03, f'{val_m[i]:.2f}', 
                     ha='center', va='top', color='white', fontsize=9)
            # Test label (at the top of the stacked bar)
            test_value = val_m[i] + diff_m[i]
            plt.text(x[i] + offset, test_value + 0.02, f'{test_value:.2f}',
                     ha='center', va='bottom', color='black', fontsize=9)

    plt.xticks(x, datasets, fontsize=12)
    plt.ylabel('Score', fontsize=14)
    plt.title('Metrics Comparison: Validation (base) + Test Difference (top)', fontsize=16)
    plt.ylim(0, 1.1)

    # Create a custom legend
    # We show the metrics in the legend. Since each metric is plotted separately,
    # we can grab the first bottom bar of each metric for a legend entry.
    # Or simply create custom handles:
    from matplotlib.patches import Patch
    legend_handles = []
    for m, metric in enumerate(metrics):
        metric_colors = colors[metric]
        legend_handles.append(Patch(facecolor=metric_colors[0], label=f'{metric} (Val)', edgecolor='none'))
    legend_handles.append(Patch(facecolor='#bbbbbb', label='Test Difference', edgecolor='none'))

    plt.legend(handles=legend_handles, loc='upper right', fontsize=5)
    
    plt.tight_layout()
    plt.savefig("All_Dataset_Combined.png", dpi=600)
    #plt.show()
    plt.close()
