import torch
import platform
import psutil
import torchmetrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from jupyterthemes import jtplot
sns.set_palette('husl', 10)
jtplot.style(theme="monokai", context="notebook", ticks=True, grid=True)
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from torch.utils.data import DataLoader


class Metrics:
    def calculate_metrics(self, preds, targets, threshold=0.5):
        # Initialize torchmetrics metrics for binary classification
        accuracy_metric = torchmetrics.Accuracy(task='binary')
        confusion_matrix_metric = torchmetrics.ConfusionMatrix(task='binary', num_classes=2)
        roc_auc_metric = torchmetrics.AUROC(task='binary')
        average_precision_metric = torchmetrics.AveragePrecision(task='binary')

        # Binarize predictions using the threshold
        preds_tensor = torch.tensor(preds)
        binarized_preds = (preds_tensor >= threshold).int()
        targets_tensor = torch.tensor(targets).int().squeeze()

        # Calculate accuracy
        self.accuracy = accuracy_metric(binarized_preds, targets_tensor).item()

        # Calculate confusion matrix
        confusion_matrix = confusion_matrix_metric(binarized_preds, targets_tensor)
        tn, fp, fn, tp = confusion_matrix.flatten()

        # Calculate precision, recall, and F1-score for each class
        precision_class_0 = tn / (tn + fp) if (tn + fp) != 0 else 0
        precision_class_1 = tp / (tp + fn) if (tp + fn) != 0 else 0
        recall_class_0 = tn / (tn + fn) if (tn + fn) != 0 else 0
        recall_class_1 = tp / (tp + fp) if (tp + fp) != 0 else 0
        f1_class_0 = 2 * (precision_class_0 * recall_class_0) / (precision_class_0 + recall_class_0) if (precision_class_0 + recall_class_0) != 0 else 0
        f1_class_1 = 2 * (precision_class_1 * recall_class_1) / (precision_class_1 + recall_class_1) if (precision_class_1 + recall_class_1) != 0 else 0

        self.precision = [float(precision_class_0), float(precision_class_1)]
        self.recall = [float(recall_class_0), float(recall_class_1)]
        self.f1 = [float(f1_class_0), float(f1_class_1)]

        # Calculate ROC AUC and Average Precision (PR AUC)
        self.roc_auc = float(roc_auc_metric(preds_tensor, targets_tensor).item())
        self.average_precision = float(average_precision_metric(preds_tensor, targets_tensor).item())

        # Display distribution of predictions and targets
        print("Distribution of predictions:", np.bincount(binarized_preds.numpy()))
        print("Distribution of targets:", np.bincount(targets_tensor.numpy()))

        print(f"Accuracy: {self.accuracy:.4f}")
        print(f"Precision (class 0): {self.precision[0]:.4f}, Precision (class 1): {self.precision[1]:.4f}")
        print(f"Recall (class 0): {self.recall[0]:.4f}, Recall (class 1): {self.recall[1]:.4f}")
        print(f"F1 Score (class 0): {self.f1[0]:.4f}, F1 Score (class 1): {self.f1[1]:.4f}")
        print(f"ROC AUC: {self.roc_auc:.4f}")
        print(f"Average Precision (PR AUC): {self.average_precision:.4f}")

    def plot_comparison(self, model_size, num_params, training_time_minutes=None):
        data = {
            'Metric': ['Accuracy', 'Precision (class 0)', 'Precision (class 1)', 'Recall (class 0)', 'Recall (class 1)', 'F1 Score (class 0)', 'F1 Score (class 1)', 'ROC AUC', 'Average Precision (PR AUC)', 'Model Size (MB)', 'Number of Parameters'],
            'xLSTM': [
                round(self.accuracy, 4),
                round(self.precision[0], 4),
                round(self.precision[1], 4),
                round(self.recall[0], 4),
                round(self.recall[1], 4),
                round(self.f1[0], 4),
                round(self.f1[1], 4),
                round(self.roc_auc, 4),
                round(self.average_precision, 4),
                round(model_size, 2),
                num_params,
            ]
        }

        if training_time_minutes is not None:
            data['Metric'].append('Training Time (m:s)')
            data['xLSTM'].append(f"{int(training_time_minutes[0])}:{int(training_time_minutes[1] * 60)}")

        # Collecting system information
        system_info = platform.uname()
        data['Metric'].extend(['Processor', 'Number of Processor Cores', 'RAM', 'GPU', 'Number of GPU Cores', 'GPU RAM', 'Platform'])
        data['xLSTM'].extend([
            system_info.processor,
            psutil.cpu_count(logical=True),
            f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU',
            torch.cuda.get_device_properties(0).multi_processor_count if torch.cuda.is_available() else 'N/A',
            f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB" if torch.cuda.is_available() else 'N/A',
            system_info.system
        ])

        df = pd.DataFrame(data)
        print(df)

        metrics_mapping = {
            'Accuracy': 'accuracy',
            'Precision (class 0)': 'precision',
            'Precision (class 1)': 'precision',
            'Recall (class 0)': 'recall',
            'Recall (class 1)': 'recall',
            'F1 Score (class 0)': 'f1',
            'F1 Score (class 1)': 'f1',
            'ROC AUC': 'roc_auc',
            'Average Precision (PR AUC)': 'average_precision'
        }

        metrics = ['Accuracy', 'Precision (class 0)', 'Precision (class 1)', 'Recall (class 0)', 'Recall (class 1)', 'F1 Score (class 0)', 'F1 Score (class 1)', 'ROC AUC', 'Average Precision (PR AUC)']
        xLSTM_values = [self.__dict__[metrics_mapping[m]][0] if '0' in m else self.__dict__[metrics_mapping[m]][1] if '1' in m else self.__dict__[metrics_mapping[m]] for m in metrics]

        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        rects = ax.bar(x, xLSTM_values, width, label='xLSTM')

        ax.set_xlabel('Metrics for xLSTM Model')
        ax.set_title('')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.grid(False)
        ax.legend()

        # Adjust the y-axis limit
        ax.set_ylim(0, 1.1)  # Set the limit a bit higher than 1 to provide space for the labels

        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f'{height:.4f}', 
                xy=(rect.get_x() + rect.get_width() / 2, height), 
                xytext=(0, 3),
                textcoords="offset points", 
                ha='center', 
                va='bottom', 
                color='red',  # Set the color to red
                fontsize=20  # Adjust the font size as needed
            )

        plt.tight_layout()
        plt.show()


        # Plot model size, number of parameters, and training time if available
        fig, axes = plt.subplots(1, 3 if training_time_minutes is not None else 2, figsize=(15, 5))

        axes[0].bar(['xLSTM'], [model_size], color=['blue'])
        axes[0].set_title('Model Size (MB)')
        axes[0].set_ylabel('Size (MB)')

        axes[1].bar(['xLSTM'], [num_params], color=['blue'])
        axes[1].set_title('Number of Parameters')
        axes[1].set_ylabel('Parameters')

        if training_time_minutes is not None:
            axes[2].bar(['xLSTM'], [training_time_minutes[0] * 60 + training_time_minutes[1] * 60], color=['blue'])
            axes[2].set_title('Training Time (s)')
            axes[2].set_ylabel('Time (s)')

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, preds, targets, threshold=0.5):
        preds_tensor = torch.tensor(preds)
        binarized_preds = (preds_tensor >= threshold).int()
        targets_tensor = torch.tensor(targets).int().squeeze()

        cm = confusion_matrix(targets_tensor.numpy(), binarized_preds.numpy())

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Hate', 'Hate'], yticklabels=['No Hate', 'Hate'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.grid(False)
        plt.show()



class PlottingMetrics:
    @staticmethod
    def plot_metrics(validation_steps, avg_train_losses, val_losses, train_accuracies=None, val_accuracies=None, avg_epoch_losses=None):
        # Determine the number of subplots needed
        subplots = []

        if avg_train_losses is not None and val_losses is not None:
            subplots.append(('Training and Validation Loss', 'Validation Steps', 'Loss', avg_train_losses, val_losses, 'Training Loss', 'Validation Loss'))
        
        if train_accuracies is not None and val_accuracies is not None:
            subplots.append(('Training and Validation Accuracy', 'Validation Steps', 'Accuracy', train_accuracies, val_accuracies, 'Training Accuracy', 'Validation Accuracy'))
        
        if avg_epoch_losses is not None:
            subplots.append(('Average Epoch Loss', 'Epochs', 'Avg Epoch Loss', avg_epoch_losses, None, 'Avg Epoch Loss', None))

        # Plot training and validation metrics
        num_subplots = len(subplots)
        fig, ax = plt.subplots(num_subplots, 1, figsize=(12, 5 * num_subplots))

        if num_subplots == 1:
            ax = [ax]

        for i, (title, xlabel, ylabel, data1, data2, label1, label2) in enumerate(subplots):
            ax[i].plot(range(1, len(data1) + 1), data1, label=label1, marker='o')
            if data2 is not None:
                ax[i].plot(range(1, len(data2) + 1), data2, label=label2, marker='o')
            ax[i].set_xlabel(xlabel)
            ax[i].set_ylabel(ylabel)
            ax[i].legend()
            ax[i].set_title(title)

        ax.grid(False)
        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_roc_pr_curves(preds, targets):
        from sklearn.metrics import roc_curve, precision_recall_curve, auc

        # Compute ROC curve and ROC area for each class
        fpr, tpr, _ = roc_curve(targets, preds)
        roc_auc = auc(fpr, tpr)

        # Compute Precision-Recall curve and area for each class
        precision, recall, _ = precision_recall_curve(targets, preds)
        pr_auc = auc(recall, precision)

        # Plot ROC curve
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        ax1.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('Receiver Operating Characteristic')
        ax1.legend(loc="lower right")

        # Plot Precision-Recall curve
        ax2.plot(recall, precision, color='blue', lw=2, label='PR curve (area = %0.2f)' % pr_auc)
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend(loc="lower left")

        plt.tight_layout()
        plt.show()


import os
from imblearn.over_sampling import ADASYN
from typing import Optional
from tqdm import tqdm
import torch
from torchmetrics import Metric
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
#from torch.utils.data import TensorDataset

# Metric for sequence accuracy with additional metrics for precision, recall, and F1 score
class SequenceMetrics(Metric):
    is_differentiable: Optional[bool] = False
    higher_is_better: Optional[bool] = True

    def __init__(self, **kwargs):
        super().__init__()
        self.accuracy = BinaryAccuracy(**kwargs)
        self.precision = BinaryPrecision(**kwargs)
        self.recall = BinaryRecall(**kwargs)
        self.f1 = BinaryF1Score(**kwargs)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = torch.round(torch.sigmoid(preds)).view(-1)
        target = target.view(-1)
        self.accuracy.update(preds, target)
        self.precision.update(preds, target)
        self.recall.update(preds, target)
        self.f1.update(preds, target)

    def compute(self):
        return {
            "accuracy": self.accuracy.compute(),
            "precision": self.precision.compute(),
            "recall": self.recall.compute(),
            "f1": self.f1.compute(),
        }

    def reset(self):
        super().reset()
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        



# Function to handle interrupt signal
def signal_handler(sig, frame):
    global interrupted
    print('Training interrupted. Saving the model...')
    interrupted = True

#   # Create balanced dataloader using imbalanced-learn
# def create_balanced_dataloader(dataloader, batch_size):
#     all_input_ids = []
#     all_labels = []

#     for batch in dataloader:
#         all_input_ids.append(batch[0])
#         all_labels.append(batch[1])

#     all_input_ids = torch.cat(all_input_ids, dim=0)
#     all_labels = torch.cat(all_labels, dim=0)
#     # Convert to numpy for imbalanced-learn
#     input_ids_np = all_input_ids.cpu().numpy()
#     labels_np = all_labels.cpu().numpy()
#     # Apply ADASYN for oversampling
#     adasyn = ADASYN(sampling_strategy=0.5, n_jobs=-1)
#     #adasyn = ADASYN()
#     balanced_input_ids_np, balanced_labels_np = adasyn.fit_resample(input_ids_np, labels_np)
#     # Convert back to tensors
#     balanced_input_ids = torch.tensor(balanced_input_ids_np)
#     balanced_labels = torch.tensor(balanced_labels_np)
#     # Create a new dataset and dataloader
#     balanced_dataset = TensorDataset(balanced_input_ids, balanced_labels)
#     balanced_dataloader = DataLoader(balanced_dataset, batch_size=batch_size, shuffle=True)

#     return balanced_dataloader

from imblearn.over_sampling import ADASYN

def create_balanced_dataloader(dataloader, batch_size):
    input_ids_list = []
    labels_list = []

    # Extract data from the DataLoader
    for batch in dataloader:
        input_ids, labels = batch
        input_ids_list.append(input_ids)
        labels_list.append(labels)

    # Concatenate all batches
    input_ids_tensor = torch.cat(input_ids_list)
    labels_tensor = torch.cat(labels_list)

    # Convert to numpy arrays
    input_ids_np = input_ids_tensor.numpy()
    labels_np = labels_tensor.numpy()

    # Reinitialize ADASYN with the appropriate sampling strategy
    adasyn = ADASYN(sampling_strategy=0.5, n_neighbors=5, n_jobs=-1)

    try:
        # Apply ADASYN to the data
        balanced_input_ids_np, balanced_labels_np = adasyn.fit_resample(input_ids_np, labels_np)
    except ValueError as e:
        print(f"ADASYN error: {e}")
        return dataloader  # Return the original dataloader if ADASYN fails

    # Convert back to tensors
    balanced_input_ids = torch.tensor(balanced_input_ids_np, dtype=torch.long)
    balanced_labels = torch.tensor(balanced_labels_np, dtype=torch.float)

    # Create a new balanced dataset
    balanced_dataset = torch.utils.data.TensorDataset(balanced_input_ids, balanced_labels)
    balanced_dataloader = DataLoader(balanced_dataset, batch_size=batch_size, shuffle=True)

    return balanced_dataloader