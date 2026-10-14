import os
import time
import argparse
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR
from sklearn.metrics import roc_auc_score, f1_score, classification_report, precision_score, recall_score, average_precision_score
from sklearn.model_selection import train_test_split
from transformers import AutoProcessor

# Import Habana modules for Gaudi support
try:
    import habana_frameworks.torch.core as htcore
    import habana_frameworks.torch.distributed as htdist
    HABANA_AVAILABLE = True
except ImportError:
    HABANA_AVAILABLE = False
    print("Habana modules not available, falling back to standard PyTorch")

# Import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WandB not available. Install with: pip install wandb")

# Import our custom modules
from dataloader import CXRDataset
from model import MedSIGLIPSlotAttention


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Weighting factor for each class. Can be:
                   - None: no weighting
                   - float: same weight for all classes
                   - Tensor of shape [num_classes]: class-specific weights
            gamma: Focusing parameter (default: 2.0)
            reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [batch_size, num_classes] logits
            targets: [batch_size] class indices
        
        Returns:
            Focal loss value
        """
        # Get probabilities
        p = F.softmax(inputs, dim=1)
        
        # Get class probabilities
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Calculate focal loss
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss



class EarlyStopping:
    """Early stopping handler to stop training when validation metric stops improving."""
    
    def __init__(self, patience=7, min_delta=0.0, mode='min', verbose=True):
        """
        Args:
            patience (int): How many epochs to wait after last improvement
            min_delta (float): Minimum change to qualify as an improvement
            mode (str): 'min' for loss (lower is better), 'max' for metrics like AUC (higher is better)
            verbose (bool): Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            raise ValueError(f"Mode {mode} is unknown, use 'min' or 'max'")
    
    def __call__(self, current_score, epoch):
        """
        Check if training should stop.
        
        Args:
            current_score: Current validation metric value
            epoch: Current epoch number
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            return False
        
        # Check if there's an improvement
        if self.monitor_op(current_score - self.min_delta, self.best_score):
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                logger.info(f"Early stopping: Validation metric improved to {current_score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"Early stopping: No improvement for {self.counter}/{self.patience} epochs "
                          f"(best: {self.best_score:.4f} at epoch {self.best_epoch + 1})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info(f"Early stopping triggered! Best score: {self.best_score:.4f} "
                              f"at epoch {self.best_epoch + 1}")
                return True
        
        return False
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0


# Set up logging
def setup_logging(rank, args):
    os.makedirs(os.path.join('/home/user/cxr_cbm/logs', args.exp_name), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join('/home/user/cxr_cbm/logs', args.exp_name, f"training_rank{rank}.log")),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train MedSIGLIP Slot Attention CXR classifier with DDP')
    parser.add_argument('--csv_file', type=str, default='/home/user/cxr_cbm/data/cxr_cbm_exps/concepts_df.csv', 
                        help='Path to CSV file with image paths and labels')
    parser.add_argument('--root_dir', type=str, default='/home/user/cxr_cbm/data/mimic_jpg_cbm', 
                        help='Root directory for images')
    parser.add_argument('--exp_name', type=str, required=True, 
                        help='Experiment name for saving checkpoints and logs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per GPU/Gaudi')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for regularization')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension for slot attention and classifiers')
    parser.add_argument('--check_freq', type=int, default=50, help='frequency to save attention maps')

    #slot params
    parser.add_argument('--slot_iters', type=int, default=3, help='Number of slot attention iterations')
    parser.add_argument('--lambda_attn_loss', type=float, default=0.01, help='Weight for attention loss')
    parser.add_argument('--lambda_competition', type=float, default=0.01, help='Weight for slot competition/diversity loss')
    parser.add_argument('--lambda_ce', type=float, default=1, help='Weight for ce loss')
    
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--train_split', type=float, default=0.8, help='Training split ratio')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.1, help='Test split ratio')
    parser.add_argument('--stratify_column', type=str, default='clear lungs_bin', 
                        help='Column to use for stratified splitting (optional)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers per GPU/Gaudi for data loading')
    parser.add_argument('--save_freq', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--use_gaudi', default=True, help='Use Habana Gaudi accelerators')
    parser.add_argument('--backend', type=str, default='hccl', help='DDP backend (nccl, gloo, hccl)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--freeze_backbone', action='store_true', default=True,
                        help='Freeze MedSIGLIP backbone (only train slot attention and classifiers)')
    parser.add_argument('--need_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='medsiglip-slot-attention', 
                        help='WandB project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='WandB run name')
    
    # Dimension reduction
    parser.add_argument('--reduced_dim', type=int, default=256, 
                        help='Reduced embedding dimension after MedSIGLIP (default: 256)')
    

    #Loss
    parser.add_argument('--loss_type', type=str, default='ce', choices=['ce', 'focal'],
                        help='Loss function to use: ce (Cross-Entropy) or focal (Focal Loss)')
    parser.add_argument('--focal_alpha', type=float, default=None,
                        help='Alpha parameter for Focal Loss (default: None, uses class weights if available)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Gamma parameter for Focal Loss (default: 2.0)')

    # Class weights
    parser.add_argument('--use_class_weights', action='store_true', default=True,
                        help='Use class weights to handle imbalance')
    
    # Attention map saving
    parser.add_argument('--save_attention_maps', action='store_true', default=True,
                        help='Save attention maps for 10 samples during test evaluation')
    parser.add_argument('--num_attention_samples', type=int, default=10,
                        help='Number of samples to save attention maps for (default: 10)')
    
    # Early stopping arguments
    parser.add_argument('--early_stopping', action='store_true', default=True,
                        help='Enable early stopping')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Number of epochs to wait before early stopping')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.0001,
                        help='Minimum change to qualify as improvement')
    parser.add_argument('--early_stopping_metric', type=str, default='loss', 
                        choices=['loss', 'auc'],
                        help='Metric to monitor for early stopping (loss or auc)')
    parser.add_argument('--embedding_path', default=None, 
                        help='Path to precomputed embeddings (optional)')
                    
    return parser.parse_args()


def convert_labels_for_loss(labels):
    """
    Convert labels from [1.0, 0, -1.0] format to [0, 1, 2] for CrossEntropyLoss
    
    Args:
        labels: Tensor with shape [batch_size, num_tasks]
    
    Returns:
        Tensor with converted labels [batch_size, num_tasks]
    """
    device = labels.device
    converted_labels = torch.zeros_like(labels, dtype=torch.long)
    converted_labels = torch.where(labels == 1.0, torch.tensor(0, device=device), converted_labels)
    converted_labels = torch.where(labels == 0.0, torch.tensor(1, device=device), converted_labels)
    converted_labels = torch.where(labels == -1.0, torch.tensor(2, device=device), converted_labels)
    return converted_labels


def convert_logits_to_probas(logits, return_all_classes=False):
    """
    Convert logits to probabilities for AUC calculation
    
    Args:
        logits: Tensor of shape [batch_size, num_tasks, num_classes]
        return_all_classes: If True, return probabilities for all classes
    
    Returns:
        Dictionary with probabilities per task
    """
    probas_dict = {}
    batch_size, num_tasks, num_classes = logits.shape
    
    for task_idx in range(num_tasks):
        # Apply softmax to get probabilities
        probs = torch.softmax(logits[:, task_idx, :], dim=1)
        
        if return_all_classes:
            # Return all class probabilities [batch_size, 3]
            probas_dict[f'disease_{task_idx}'] = probs.cpu().detach().numpy()
        else:
            # Get probability of positive class (index 0 corresponds to 1.0)
            probas_dict[f'disease_{task_idx}'] = probs[:, 0].cpu().detach().numpy()
    
    return probas_dict


def convert_logits_to_predictions(logits):
    """
    Convert logits to class predictions
    
    Args:
        logits: Tensor of shape [batch_size, num_tasks, num_classes]
    
    Returns:
        Dictionary with class predictions per task
    """
    preds_dict = {}
    batch_size, num_tasks, num_classes = logits.shape
    
    for task_idx in range(num_tasks):
        # Get the class with highest probability
        preds = torch.argmax(logits[:, task_idx, :], dim=1).cpu().detach().numpy()
        preds_dict[f'disease_{task_idx}'] = preds
    
    return preds_dict


def calculate_class_weights(train_loader, disease_names, device):
    """
    Calculate class weights for each disease based on training data.
    
    Args:
        train_loader: DataLoader for training set
        disease_names: List of disease names
        device: Device to place weights on
    
    Returns:
        Dictionary mapping disease index to class weight tensors
        Also returns tensor of shape [num_diseases, 3] for efficient batching
    """
    logger.info("Calculating class weights from training data...")
    
    # Collect all labels for each disease
    num_diseases = len(disease_names)
    disease_labels_dict = {i: [] for i in range(num_diseases)}
    
    for batch in tqdm(train_loader, desc="Collecting labels for class weights"):
        labels = batch['labels']  # Shape: [batch_size, num_diseases]
        
        for i in range(num_diseases):
            disease_labels = labels[:, i]
            # Convert to class indices [0, 1, 2]
            disease_labels_converted = convert_labels_for_loss(disease_labels.unsqueeze(1)).squeeze()
            disease_labels_dict[i].extend(disease_labels_converted.tolist())
    
    # Calculate weights for each disease
    class_weights_dict = {}
    all_weights_list = []
    
    for i, disease_name in enumerate(disease_names):
        disease_labels_tensor = torch.tensor(disease_labels_dict[i], dtype=torch.long)
        
        # Count instances of each class
        class_counts = torch.bincount(disease_labels_tensor, minlength=3)
        
        # Calculate weights: total_samples / (num_classes * count_of_class)
        total_samples = len(disease_labels_tensor)
        num_classes = 3
        
        # Avoid division by zero
        weights = total_samples / (num_classes * (class_counts.float() + 1e-6))
        
        # Normalize weights so that they sum to num_classes
        weights = weights / weights.sum() * num_classes
        
        class_weights_dict[i] = weights.to(device)
        all_weights_list.append(weights)
        
        # Log only first 5 and last 5 diseases to avoid clutter
        if i < 5 or i >= num_diseases - 5:
            logger.info(f"{disease_name} (task {i}):")
            logger.info(f"  Class counts: {class_counts.tolist()}")
            logger.info(f"  Class weights: {weights.tolist()}")
        elif i == 5:
            logger.info(f"... (logging suppressed for tasks 5-{num_diseases-6}) ...")
    
    # Stack all weights into a tensor for efficient use: [num_diseases, 3]
    all_weights_tensor = torch.stack(all_weights_list).to(device)
    
    logger.info(f"\nClass weights calculated for {num_diseases} diseases")
    logger.info(f"Weights tensor shape: {all_weights_tensor.shape}")
    
    return class_weights_dict, all_weights_tensor


def calculate_weighted_ce_loss(logits, labels, class_weights_tensor):
    """
    Calculate weighted cross-entropy loss for all tasks efficiently.
    
    Args:
        logits: [batch_size, num_tasks, num_classes]
        labels: [batch_size, num_tasks] with class indices
        class_weights_tensor: [num_tasks, num_classes] - weights for each task
    
    Returns:
        Average weighted CE loss across all tasks
    """
    batch_size, num_tasks, num_classes = logits.shape
    
    total_loss = 0.0
    
    for task_idx in range(num_tasks):
        # Get weights for this task: [num_classes]
        task_weights = class_weights_tensor[task_idx]
        
        # Calculate weighted CE loss for this task
        task_loss = F.cross_entropy(
            logits[:, task_idx, :],  # [batch_size, num_classes]
            labels[:, task_idx],     # [batch_size]
            weight=task_weights,
            # label_smoothing=0.005
        )
        
        total_loss += task_loss
    
    # Average across tasks
    return total_loss / num_tasks



def calculate_weighted_focal_loss(logits, labels, focal_loss_fn, class_weights_tensor=None):
    """
    Calculate focal loss for all tasks efficiently.
    
    Args:
        logits: [batch_size, num_tasks, num_classes]
        labels: [batch_size, num_tasks] with class indices
        focal_loss_fn: FocalLoss instance
        class_weights_tensor: [num_tasks, num_classes] - weights for each task (optional)
    
    Returns:
        Average focal loss across all tasks
    """
    batch_size, num_tasks, num_classes = logits.shape
    
    total_loss = 0.0
    
    for task_idx in range(num_tasks):
        # Get class weights for this task if available
        if class_weights_tensor is not None:
            task_weights = class_weights_tensor[task_idx]
            # Create a temporary focal loss with these weights
            task_focal_loss = FocalLoss(alpha=task_weights, gamma=focal_loss_fn.gamma, 
                                       reduction=focal_loss_fn.reduction)
            task_loss = task_focal_loss(logits[:, task_idx, :], labels[:, task_idx])
        else:
            task_loss = focal_loss_fn(logits[:, task_idx, :], labels[:, task_idx])
        
        total_loss += task_loss
    
    # Average across tasks
    return total_loss / num_tasks



def convert_labels_to_binary(labels):
    """
    Convert labels from [1.0, 0, -1.0] format to binary [0, 1] by excluding uncertain class
    
    Args:
        labels: numpy array with shape [batch_size, num_tasks]
    
    Returns:
        Dictionary with:
            - 'labels': numpy array with binary labels [0, 1] or None for uncertain
            - 'mask': boolean mask for valid (non-uncertain) samples
    """
    binary_labels = np.zeros_like(labels, dtype=int)
    # Positive class -> 1, Negative class -> 0, Uncertain -> -1 (will be masked)
    binary_labels[labels == 1.0] = 1
    binary_labels[labels == -1.0] = 0
    binary_labels[labels == 0.0] = -1  # Mark uncertain
    
    mask = labels != 0.0  # Mask for non-uncertain samples
    
    return {'labels': binary_labels, 'mask': mask}


def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, args, is_best=False, rank=0):
    """Save model checkpoint - only save best model"""
    if rank != 0:
        return
    
    os.makedirs(os.path.join(args.checkpoint_dir, args.exp_name), exist_ok=True)
    
    best_path = os.path.join(args.checkpoint_dir, args.exp_name, "best_model.pt")
    
    # For DDP, save the model's module state_dict
    if isinstance(model, DDP):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    
    state = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_loss': best_val_loss,
    }
    
    if is_best:
        torch.save(state, best_path)
        logger.info(f"Best model saved to {best_path} (epoch {epoch+1})")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """Load model checkpoint"""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # For DDP, load to module
    if isinstance(model, DDP):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    


    # Move optimizer states to the right device
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    
    return model, optimizer, scheduler, start_epoch, best_val_loss


def load_checkpoint_for_eval(model, checkpoint_path, device):
    """Load model checkpoint for evaluation only"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # For DDP, load to module
    if isinstance(model, DDP):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    epoch = checkpoint['epoch']
    return model, epoch




def calculate_aucs(all_labels, all_probas, disease_names, exclude_uncertain=False):
    """
    Calculate AUC for each disease as a multi-class problem
    
    Args:
        all_labels: numpy array of labels
        all_probas: dict of probabilities
        disease_names: list of disease names
        exclude_uncertain: if True, calculate binary AUC excluding uncertain class
    
    Returns:
        Dictionary of AUC scores per disease
    """
    aucs = {}
    
    for i, disease_name in enumerate(disease_names):
        disease_key = f'disease_{i}'
        if disease_key in all_probas:
            # Get labels for this disease (values: 1.0, 0.0, -1.0)
            disease_labels = all_labels[:, i]
            
            # Get probability predictions
            probas = all_probas[disease_key]
            if isinstance(probas, list):
                probas = np.array(probas)
            
            if exclude_uncertain:
                # Binary classification: exclude uncertain samples
                binary_data = convert_labels_to_binary(disease_labels.reshape(-1, 1))
                binary_labels = binary_data['labels'].flatten()
                mask = binary_data['mask'].flatten()
                
                # Filter out uncertain samples
                if mask.sum() < 2:  # Need at least 2 samples
                    continue
                
                filtered_labels = binary_labels[mask]
                
                # Check if we have both classes
                unique_labels = np.unique(filtered_labels)
                if len(unique_labels) < 2:
                    logger.warning(f"Only {len(unique_labels)} class(es) present for {disease_name} (binary), skipping AUC")
                    continue
                
                # Get positive class probability (class 0 = positive in our 3-class output)
                if len(probas.shape) == 1:
                    # Already 1D probabilities
                    filtered_probas = probas[mask]
                else:
                    # probas is [batch_size, 3]
                    # For binary: class 0 (positive) -> label 1, class 2 (negative) -> label 0
                    # So we want probability of class 0 (positive)

                    # pos_proba = probas[:, 0]
                    # filtered_probas = pos_proba[mask]

                    pos_probas = probas[mask, 0]
                    uncertain_probas = probas[mask, 1]
                    neg_probas = probas[mask, 2]
                    filtered_probas = pos_probas / (pos_probas + uncertain_probas + neg_probas + 1e-8)
                    
                
                try:
                    auc = roc_auc_score(filtered_labels, filtered_probas)
                    if not np.isnan(auc) and not np.isinf(auc):
                        aucs[disease_name] = auc
                except Exception as e:
                    logger.warning(f"Could not calculate binary AUC for {disease_name}: {e}")
            
            else:
                # Multi-class AUC with all three classes
                # Convert labels to class indices [0, 1, 2]
                label_indices = np.zeros_like(disease_labels, dtype=int)
                label_indices[disease_labels == 1.0] = 0
                label_indices[disease_labels == 0.0] = 1
                label_indices[disease_labels == -1.0] = 2
                
                # Check probas shape
                if len(probas.shape) == 1:
                    logger.warning(f"Probabilities for {disease_name} are 1D, skipping AUC calculation")
                    continue
                
                # Check if we have valid probabilities
                if np.any(np.isnan(probas)) or np.any(np.isinf(probas)):
                    logger.warning(f"Invalid probabilities detected for {disease_name}, skipping")
                    continue
                
                # Check which classes are present
                unique_labels = np.unique(label_indices)
                
                # For multi-class AUC, we need at least 2 classes
                if len(unique_labels) < 2:
                    logger.warning(f"Only {len(unique_labels)} class(es) present for {disease_name}, skipping AUC")
                    continue
                
                try:
                    # For multi-class AUC, we need probabilities for all classes present
                    # Extract only the columns for classes that are present
                    probas_filtered = probas[:, unique_labels]
                    
                    # Calculate multi-class AUC using one-vs-rest strategy
                    if len(unique_labels) == 2:
                        # Map labels to 0 and 1 for binary AUC
                        binary_labels = np.zeros_like(label_indices)
                        binary_labels[label_indices == unique_labels[1]] = 1
                        
                        # Use probability of the second class (class 1 in binary)
                        # This is the standard approach for binary AUC
                        if probas_filtered.shape[1] == 2:
                            auc = roc_auc_score(binary_labels, probas_filtered[:, 1])
                        else:
                            # Fallback: use first column if shape is unexpected
                            auc = roc_auc_score(binary_labels, probas_filtered[:, 0])
                    else:
                        # True multi-class (3 classes): use one-vs-rest strategy
                        auc = roc_auc_score(label_indices, probas_filtered, 
                                           multi_class='ovr', 
                                           average='macro',
                                           labels=list(unique_labels))
                    
                    # Check if AUC is valid
                    if not np.isnan(auc) and not np.isinf(auc):
                        aucs[disease_name] = auc
                    else:
                        logger.warning(f"Invalid AUC value for {disease_name}: {auc}")
                        
                except Exception as e:
                    logger.warning(f"Could not calculate AUC for {disease_name}: {e}")
    
    return aucs




def calculate_f1_scores(all_labels, all_preds, disease_names, exclude_uncertain=False):
    f1_scores = {}
    precision_scores_dict = {}
    recall_scores_dict = {}
    
    for i, disease_name in enumerate(disease_names):
        disease_key = f'disease_{i}'
        if disease_key in all_preds:
            # 1. Prepare Ground Truth Indices (mapping: 1.0->0, 0.0->1, -1.0->2)
            disease_labels = all_labels[:, i]
            label_indices = np.zeros_like(disease_labels, dtype=int)
            label_indices[disease_labels == 1.0] = 0
            label_indices[disease_labels == 0.0] = 1
            label_indices[disease_labels == -1.0] = 2
            
            # 2. Prepare Prediction Indices (already 0, 1, 2)
            preds_array = np.array(all_preds[disease_key])
            
            if exclude_uncertain:
                # --- NEW LOGIC: Filter only by Ground Truth ---
                # Mask: Keep only Positive (1.0) and Negative (-1.0) ground truths
                gt_mask = (disease_labels == 1.0) | (disease_labels == -1.0)
                
                if gt_mask.sum() < 2:
                    continue
                
                # Filter both arrays using the GT mask
                # We keep the predictions as they are (even if they predicted 'uncertain')
                filtered_labels = label_indices[gt_mask]
                filtered_preds = preds_array[gt_mask]
                
                # Check how many classes are actually present in the filtered set
                unique_labels = np.unique(filtered_labels)
                if len(unique_labels) < 2:
                    continue

                try:
                    # Calculate 3-class macro metrics on the filtered data
                    # We specify labels=[0, 1, 2] to ensure the macro average 
                    # accounts for the potential to predict 'uncertain' (1)
                    f1 = f1_score(filtered_labels, filtered_preds, average='macro', labels=[0, 1, 2], zero_division=0)
                    prec = precision_score(filtered_labels, filtered_preds, average='macro', labels=[0, 1, 2], zero_division=0)
                    rec = recall_score(filtered_labels, filtered_preds, average='macro', labels=[0, 1, 2], zero_division=0)
                    
                    f1_scores[disease_name] = f1
                    precision_scores_dict[disease_name] = prec
                    recall_scores_dict[disease_name] = rec
                except Exception as e:
                    logger.warning(f"Could not calculate filtered 3-class F1 for {disease_name}: {e}")
            
            else:
                # Standard Multi-class metrics (All samples)
                try:
                    f1_macro = f1_score(label_indices, preds_array, average='macro', zero_division=0)
                    precision_macro = precision_score(label_indices, preds_array, average='macro', zero_division=0)
                    recall_macro = recall_score(label_indices, preds_array, average='macro', zero_division=0)

                    f1_scores[disease_name] = f1_macro
                    precision_scores_dict[disease_name] = precision_macro
                    recall_scores_dict[disease_name] = recall_macro
                except Exception as e:
                    logger.warning(f"Could not calculate F1 for {disease_name}: {e}")
    
    return f1_scores, precision_scores_dict, recall_scores_dict



def calculate_maps(all_labels, all_probas, disease_names, exclude_uncertain=False):
    """
    Calculate Average Precision (AP) for each disease
    
    Args:
        all_labels: numpy array of labels [N, num_diseases]
        all_probas: dict of probabilities {disease_i: [N, 3]}
        disease_names: list of disease names
        exclude_uncertain: if True, calculate binary AP excluding uncertain class
    
    Returns:
        Dictionary of AP scores per disease
    """
    aps = {}
    
    for i, disease_name in enumerate(disease_names):
        disease_key = f'disease_{i}'
        if disease_key in all_probas:
            # Get labels for this disease (values: 1.0, 0.0, -1.0)
            disease_labels = all_labels[:, i]
            
            # Get probability predictions [N, 3]
            probas = all_probas[disease_key]
            if isinstance(probas, list):
                probas = np.array(probas)
            
            if exclude_uncertain:
                # --- BINARY AP (Positive vs Negative) ---
                # Mask out the uncertain samples (label 0.0)
                mask = (disease_labels == 1.0) | (disease_labels == -1.0)
                
                if mask.sum() < 2:
                    continue
                
                # Filter labels: map 1.0 -> 1 (pos), -1.0 -> 0 (neg)
                filtered_labels = (disease_labels[mask] == 1.0).astype(int)
                
                # Check if we have both classes
                if len(np.unique(filtered_labels)) < 2:
                    continue

                # For binary AP, we need the score of the positive class.
                # In your 3-class setup: Class 0=Positive, 1=Uncertain, 2=Negative
                # We re-normalize the positive vs negative probabilities
                pos_probas = probas[mask, 0]
                uncertain_probas = probas[mask, 1]
                neg_probas = probas[mask, 2]
                binary_scores = pos_probas / (pos_probas + uncertain_probas + neg_probas + 1e-8)
                
                
                try:
                    ap = average_precision_score(filtered_labels, binary_scores)
                    if not np.isnan(ap):
                        aps[disease_name] = ap
                except Exception as e:
                    logger.warning(f"Could not calculate binary AP for {disease_name}: {e}")
            
            else:
                # --- MULTI-CLASS AP (3-Class) ---
                # Convert labels to one-hot encoding for the 3 classes
                # Class mapping: 0: Positive (1.0), 1: Uncertain (0.0), 2: Negative (-1.0)
                y_true = np.zeros((disease_labels.size, 3))
                y_true[disease_labels == 1.0, 0] = 1
                y_true[disease_labels == 0.0, 1] = 1
                y_true[disease_labels == -1.0, 2] = 1
                
                try:
                    # Calculate Macro AP: average AP across the 3 classes
                    ap = average_precision_score(y_true, probas, average='macro')
                    if not np.isnan(ap):
                        aps[disease_name] = ap
                except Exception as e:
                    logger.warning(f"Could not calculate multi-class AP for {disease_name}: {e}")
                    
    return aps


def create_stratified_splits(dataframe, train_split, val_split, test_split, stratify_column, seed):
    """
    Create stratified train/val/test splits based on a specific column.
    
    Args:
        dataset: CXRDataset instance
        train_split: Proportion for training set
        val_split: Proportion for validation set
        test_split: Proportion for test set
        stratify_column: Column name to use for stratification (optional)
        seed: Random seed
    
    Returns:
        train_indices, val_indices, test_indices
    """
    # Get the dataframe
    df = dataframe
    # df = df.sample(200)
    indices = np.arange(len(df))
    
    # Use stratification if column specified
    stratify_labels = None
    if stratify_column and stratify_column in df.columns:
        stratify_labels = df[stratify_column].values
        logger.info(f"Using stratified split on column: {stratify_column}")
    else:
        logger.info("Using random split (no stratification)")
    
    # First split: separate out test set
    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=test_split,
        stratify=stratify_labels,
        random_state=seed
    )
    
    # Second split: separate train and val from remaining data
    if stratify_labels is not None:
        train_val_labels = stratify_labels[train_val_indices]
    else:
        train_val_labels = None
        
    relative_val_size = val_split / (train_split + val_split)
    
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=relative_val_size,
        stratify=train_val_labels,
        random_state=seed
    )
    
    logger.info(f"Dataset splits:")
    logger.info(f"  Train: {len(train_indices)} samples")
    logger.info(f"  Val: {len(val_indices)} samples")
    logger.info(f"  Test: {len(test_indices)} samples")
    
    return train_indices, val_indices, test_indices


def create_collate_fn(processor):
    """
    Factory function to create a collate_fn with processor in closure.
    """
    def collate_fn(batch):
        """
        Custom collate function to preprocess images with MedSIGLIP processor.
        
        Args:
            batch: List of dicts with 'image' (PIL Image) and 'labels' (tensor)
            
        Returns:
            Dict with 'image' (preprocessed pixel_values) and 'labels' (stacked tensor)
        """
        images_list = []
        labels_list = []
        
        for item in batch:
            img = item['image']  # Should be PIL Image
            
            # Validate it's a PIL Image
            if not isinstance(img, Image.Image):
                raise TypeError(f"Expected PIL Image, got {type(img)}. "
                              "Dataset should return PIL Images, not tensors.")
            
            images_list.append(img)
            labels_list.append(item['labels'])
        
        # Process all images at once with the processor
        processed = processor(images=images_list, return_tensors="pt")
        pixel_values = processed.pixel_values  # Shape: [batch_size, 3, 448, 448]
        
        # Stack labels
        labels_tensor = torch.stack(labels_list)  # Shape: [batch_size, num_diseases]
        
        return {
            'image': pixel_values,
            'labels': labels_tensor
        }
    
    return collate_fn


def evaluate_on_test(model, test_loader, device, disease_names, save_attention_maps=False, 
                     save_dir=None, num_samples_to_save=10):
    """
    Evaluate model on test set with optional attention map saving for limited samples
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for test set
        device: Device to run evaluation on
        disease_names: List of disease names
        save_attention_maps: Whether to save attention visualizations
        save_dir: Directory to save attention maps
        num_samples_to_save: Number of samples to save attention maps for
    
    Returns:
        Dictionary with AUC and F1 scores
    """
    model.eval()
    all_labels = []
    all_probas = {f'disease_{i}': [] for i in range(len(disease_names))}
    all_preds = {f'disease_{i}': [] for i in range(len(disease_names))}
    
    # Create save directory if needed
    if save_attention_maps and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Saving attention maps for {num_samples_to_save} samples to: {save_dir}")
    
    sample_idx = 0  # Global sample counter across batches
    saved_samples = 0  # Counter for saved attention maps
    
    with torch.no_grad():
        for batch_num, batch in enumerate(tqdm(test_loader, desc="Evaluating on test set")):
            images = batch['image'].to(device)
            labels = batch['labels'].to(device)
            
            # Store labels
            all_labels.append(labels.cpu().numpy())
            
            # Forward pass (no labels for inference)
            outputs = model(images)
            logits = outputs['logits']  # [batch, num_tasks, 3]
            attentions = outputs['attentions']  # List of [batch, num_patches] tensors
            
            # Get probabilities for AUC (all classes)
            probas_dict = convert_logits_to_probas(logits, return_all_classes=True)
            for disease_key, probs in probas_dict.items():
                all_probas[disease_key].append(probs)
            
            # Get predictions for F1
            preds_dict = convert_logits_to_predictions(logits)
            for disease_key, preds in preds_dict.items():
                all_preds[disease_key].extend(preds)
            
            # Save attention maps if requested and haven't reached limit
            if save_attention_maps and save_dir and saved_samples < num_samples_to_save:
                batch_size = images.shape[0]
                num_patches = attentions[0].shape[-1]
                patch_grid_size = int(np.sqrt(num_patches))
                
                for b in range(batch_size):
                    if saved_samples >= num_samples_to_save:
                        break
                    
                    # Create directory for this sample
                    sample_dir = os.path.join(save_dir, f'sample_{sample_idx:05d}')
                    os.makedirs(sample_dir, exist_ok=True)
                    
                    # Get ground truth labels for this sample
                    gt_labels = labels[b].cpu().numpy()  # [num_tasks]
                    
                    # Save attention map for each disease/task
                    for task_idx in range(len(disease_names)):
                        disease_name = disease_names[task_idx]
                        
                        # Get attention map for this task: [num_patches]
                        attn = attentions[b, task_idx].cpu().numpy()
                        
                        # Reshape to 2D grid
                        attn_2d = attn.reshape(patch_grid_size, patch_grid_size)
                        
                        # Normalize to 0-255 range
                        attn_normalized = (attn_2d - attn_2d.min()) / (attn_2d.max() - attn_2d.min() + 1e-8)
                        attn_uint8 = (attn_normalized * 255).astype(np.uint8)
                        
                        # Create PIL image and resize to 448x448

                        cmap = plt.get_cmap('jet')
                        attn_colored = cmap(attn_uint8 / 255.0)  # Normalize to [0,1]
                        attn_colored = (attn_colored[:, :, :3] * 255).astype(np.uint8)

                        attn_img = Image.fromarray(attn_colored)
                        attn_img_resized = attn_img.resize((448, 448), Image.BILINEAR)

                        # attn_img = Image.fromarray(attn_uint8, mode='L')
                        # attn_img_resized = attn_img.resize((448, 448), Image.BILINEAR)
                        
                        # Get prediction info
                        pred_class = torch.argmax(logits[b, task_idx]).item()
                        pred_prob = torch.softmax(logits[b, task_idx], dim=0)[pred_class].item()
                        
                        # Convert ground truth and prediction to labels
                        gt_value = gt_labels[task_idx]
                        if gt_value == 1.0:
                            gt_label = "positive"
                        elif gt_value == 0.0:
                            gt_label = "uncertain"
                        else:  # -1.0
                            gt_label = "negative"
                        
                        if gt_label == "uncertain":
                            # Skip saving uncertain samples to reduce clutter
                            continue

                        if pred_class == 0:
                            pred_label = "positive"
                        elif pred_class == 1:
                            pred_label = "uncertain"
                        else:  # 2
                            pred_label = "negative"
                        
                        # Create filename with disease name
                        # Clean disease name for filename (remove special characters)
                        clean_disease_name = disease_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
                        filename = f'{clean_disease_name}_gt_{gt_label}_pred_{pred_label}_prob_{pred_prob:.3f}.png'
                        
                        # Save attention map
                        save_path = os.path.join(sample_dir, filename)
                        attn_img_resized.save(save_path)
                    
                    saved_samples += 1
                    sample_idx += 1
            else:
                sample_idx += images.shape[0]
    
    if save_attention_maps and save_dir:
        logger.info(f"Saved attention maps for {saved_samples} test samples")
    
    # Concatenate all labels
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Concatenate probabilities for each disease
    for i in range(len(disease_names)):
        disease_key = f'disease_{i}'
        if disease_key in all_probas and len(all_probas[disease_key]) > 0:
            all_probas[disease_key] = np.concatenate(all_probas[disease_key], axis=0)
    


    

    # Calculate metrics WITH uncertain class (3-class)
    aucs_3class = calculate_aucs(all_labels, all_probas, disease_names, exclude_uncertain=False)
    f1_scores_3class, precision_scores_3class, recall_scores_3class = calculate_f1_scores(
        all_labels, all_preds, disease_names, exclude_uncertain=False
    )
    
    # Calculate metrics WITHOUT uncertain class (binary)
    aucs_binary = calculate_aucs(all_labels, all_probas, disease_names, exclude_uncertain=True)
    f1_scores_binary, precision_scores_binary, recall_scores_binary = calculate_f1_scores(
        all_labels, all_preds, disease_names, exclude_uncertain=True
    )
    
    # Calculate average metrics for both cases
    avg_auc_3class = np.mean(list(aucs_3class.values())) if aucs_3class else 0.0
    avg_f1_3class = np.mean(list(f1_scores_3class.values())) if f1_scores_3class else 0.0
    avg_precision_3class = np.mean(list(precision_scores_3class.values())) if precision_scores_3class else 0.0
    avg_recall_3class = np.mean(list(recall_scores_3class.values())) if recall_scores_3class else 0.0
    
    avg_auc_binary = np.mean(list(aucs_binary.values())) if aucs_binary else 0.0
    avg_f1_binary = np.mean(list(f1_scores_binary.values())) if f1_scores_binary else 0.0
    avg_precision_binary = np.mean(list(precision_scores_binary.values())) if precision_scores_binary else 0.0
    avg_recall_binary = np.mean(list(recall_scores_binary.values())) if recall_scores_binary else 0.0
    

    aps_3class = calculate_maps(all_labels, all_probas, disease_names, exclude_uncertain=False)
    avg_map_3class = np.mean(list(aps_3class.values())) if aps_3class else 0.0

    # 2. Calculate APs for binary (Pos vs Neg)
    aps_binary = calculate_maps(all_labels, all_probas, disease_names, exclude_uncertain=True)
    avg_map_binary = np.mean(list(aps_binary.values())) if aps_binary else 0.0


    return {
        # 3-class metrics
        'aucs_3class': aucs_3class,
        'f1_scores_3class': f1_scores_3class,
        'precision_scores_3class': precision_scores_3class,
        'recall_scores_3class': recall_scores_3class,
        'avg_auc_3class': avg_auc_3class,
        'avg_f1_3class': avg_f1_3class,
        'avg_precision_3class': avg_precision_3class,
        'avg_recall_3class': avg_recall_3class,
        'map_3class': avg_map_3class,
        # Binary metrics (excluding uncertain)
        'aucs_binary': aucs_binary,
        'f1_scores_binary': f1_scores_binary,
        'precision_scores_binary': precision_scores_binary,
        'recall_scores_binary': recall_scores_binary,
        'avg_auc_binary': avg_auc_binary,
        'avg_f1_binary': avg_f1_binary,
        'avg_precision_binary': avg_precision_binary,
        'avg_recall_binary': avg_recall_binary,
        'map_binary': avg_map_binary,
    }




def evaluate_best_checkpoint(args, test_loader, device, disease_names, num_tasks, save_attention_maps=False):
    """
    Evaluate the best checkpoint on the test set
    
    Args:
        save_attention_maps: If True, save attention maps for limited samples
    """
    logger.info("=" * 80)
    logger.info("Starting evaluation of best checkpoint on test set")
    logger.info("=" * 80)
    
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.exp_name)
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")

    best_checkpoint_path = "/home/user/cxr_cbm/data/cxr_cbm_exps/checkpoints/slot_attn_final/best_model.pt"
    
    if not os.path.exists(best_checkpoint_path):
        logger.warning("Best checkpoint not found!")
        return None
    
    logger.info(f"Evaluating checkpoint: best_model.pt")
    
    # Create a fresh model
    model = MedSIGLIPSlotAttention(
        num_tasks=num_tasks,
        num_classes_per_task=3,
        hidden_dim=args.hidden_dim,
        slot_iters=args.slot_iters,
        freeze_backbone=args.freeze_backbone,
        lambda_attn_loss=args.lambda_attn_loss,
        reduced_dim=args.reduced_dim,
        disease_embeddings_path=args.embedding_path
    )
    model = model.to(device)
    
    # Load checkpoint
    model, epoch = load_checkpoint_for_eval(model, best_checkpoint_path, device)
    
    attention_save_dir = None
    if save_attention_maps:
        attention_save_dir = os.path.join(checkpoint_dir, "attention_maps_test_set")
        logger.info(f"Will save attention maps for {args.num_attention_samples} samples to: {attention_save_dir}")
    
    # Evaluate on test set
    metrics = evaluate_on_test(
        model, test_loader, device, disease_names,
        save_attention_maps=save_attention_maps,
        save_dir=attention_save_dir,
        num_samples_to_save=args.num_attention_samples
    )
    
    # Store results
    result = {
        'checkpoint': 'best_model.pt',
        'epoch': epoch,
        'avg_auc_3class': metrics['avg_auc_3class'],
        'avg_f1_3class': metrics['avg_f1_3class'],
        'avg_precision_3class': metrics['avg_precision_3class'],
        'avg_recall_3class': metrics['avg_recall_3class'],
        'map_3class': metrics['map_3class'],
        'avg_auc_binary': metrics['avg_auc_binary'],
        'avg_f1_binary': metrics['avg_f1_binary'],
        'avg_precision_binary': metrics['avg_precision_binary'],
        'avg_recall_binary': metrics['avg_recall_binary'],
        'map_binary': metrics['map_binary']
    }
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SET EVALUATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Best Checkpoint (Epoch {epoch + 1}):")
    logger.info("\nMetrics WITH Uncertain Class (3-class):")
    logger.info(f"  Avg AUC: {metrics['avg_auc_3class']:.4f}")
    logger.info(f"  Avg F1: {metrics['avg_f1_3class']:.4f}")
    logger.info(f"  Avg Precision: {metrics['avg_precision_3class']:.4f}")
    logger.info(f"  Avg Recall: {metrics['avg_recall_3class']:.4f}")
    logger.info(f"  Avg MAP: {metrics['map_3class']:.4f}")
    
    logger.info("\nMetrics WITHOUT Uncertain Class (Binary):")
    logger.info(f"  Avg AUC: {metrics['avg_auc_binary']:.4f}")
    logger.info(f"  Avg F1: {metrics['avg_f1_binary']:.4f}")
    logger.info(f"  Avg Precision: {metrics['avg_precision_binary']:.4f}")
    logger.info(f"  Avg Recall: {metrics['avg_recall_binary']:.4f}")
    logger.info(f"  Avg MAP: {metrics['map_binary']:.4f}")
    
    # Log top 10 diseases by AUC for both cases
    logger.info("\nTop 10 Diseases by AUC (3-class):")
    sorted_aucs_3class = sorted(metrics['aucs_3class'].items(), key=lambda x: x[1], reverse=True)[:10]
    for disease, auc in sorted_aucs_3class:
        logger.info(f"  {disease}: {auc:.4f}")
    
    logger.info("\nTop 10 Diseases by AUC (Binary):")
    sorted_aucs_binary = sorted(metrics['aucs_binary'].items(), key=lambda x: x[1], reverse=True)[:10]
    for disease, auc in sorted_aucs_binary:
        logger.info(f"  {disease}: {auc:.4f}")
    
    # Save results to CSV
    results_df = pd.DataFrame([result])
    results_csv_path = os.path.join(checkpoint_dir, "test_evaluation_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    logger.info(f"\nResults saved to: {results_csv_path}")
    
    # Log to WandB if enabled
    if args.need_wandb and WANDB_AVAILABLE:
        wandb_table = wandb.Table(dataframe=results_df)
        wandb.log({
            "test_evaluation_results": wandb_table,
            "test_avg_auc_3class": metrics['avg_auc_3class'],
            "test_avg_f1_3class": metrics['avg_f1_3class'],
            "test_avg_auc_binary": metrics['avg_auc_binary'],
            "test_avg_f1_binary": metrics['avg_f1_binary'],
        })
        
        # Log sample attention maps if saved
        if save_attention_maps and attention_save_dir:
            # Find first sample directory
            sample_dirs = sorted(glob(os.path.join(attention_save_dir, "sample_*")))
            if sample_dirs:
                first_sample_dir = sample_dirs[0]
                # Get first few attention maps
                attn_images = sorted(glob(os.path.join(first_sample_dir, "*.png")))[:10]
                if attn_images:
                    wandb_images = [wandb.Image(img, caption=os.path.basename(img)) for img in attn_images]
                    wandb.log({"sample_attention_maps": wandb_images})
    
    logger.info("=" * 80)
    
    return results_df



def periodic_save(args, test_loader, device, disease_names, num_tasks, epoch, num_samples_to_save=10):
    logger.info("=" * 80)
    logger.info(f"Saving attention maps at {epoch}")
    logger.info("=" * 80)
    
    sample_idx = 0  # Global sample counter across batches
    saved_samples = 0  # Counter for saved attention maps

    checkpoint_dir = os.path.join(args.checkpoint_dir, args.exp_name)
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")

    if not os.path.exists(best_checkpoint_path):
        logger.warning("Best checkpoint not found!")
        return None
    
    new_model = MedSIGLIPSlotAttention(
        num_tasks=num_tasks,
        num_classes_per_task=3,
        hidden_dim=args.hidden_dim,
        slot_iters=args.slot_iters,
        freeze_backbone=args.freeze_backbone,
        lambda_attn_loss=args.lambda_attn_loss,
        reduced_dim=args.reduced_dim,
        disease_embeddings_path=args.embedding_path
    )
    new_model = new_model.to(device)
    
    # Load checkpoint
    new_model, _ = load_checkpoint_for_eval(new_model, best_checkpoint_path, device)

    save_dir = os.path.join(checkpoint_dir, f"attention_maps_epoch_{epoch+1}")
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Will save attention maps for {args.num_attention_samples} samples to: {save_dir}")

    with torch.no_grad():
        for batch_num, batch in enumerate(tqdm(test_loader, desc="Saving the heatmaps")):
            images = batch['image'].to(device)
            labels = batch['labels'].to(device)
            
            
            
            # Forward pass (no labels for inference)
            outputs = new_model(images)
            logits = outputs['logits']  # [batch, num_tasks, 3]
            attentions = outputs['attentions']  # List of [batch, num_patches] tensors
            
            # Save attention maps if requested and haven't reached limit
            if save_dir and saved_samples < num_samples_to_save:
                batch_size = images.shape[0]
                num_patches = attentions[0].shape[-1]
                patch_grid_size = int(np.sqrt(num_patches))
                
                for b in range(batch_size):
                    if saved_samples >= num_samples_to_save:
                        break
                    
                    # Create directory for this sample
                    sample_dir = os.path.join(save_dir, f'sample_{sample_idx:05d}')
                    os.makedirs(sample_dir, exist_ok=True)
                    
                    # Get ground truth labels for this sample
                    gt_labels = labels[b].cpu().numpy()  # [num_tasks]
                    
                    # Save attention map for each disease/task
                    for task_idx in range(len(disease_names)):
                        disease_name = disease_names[task_idx]
                        
                        # Get attention map for this task: [num_patches]
                        attn = attentions[b, task_idx].cpu().numpy()
                        
                        # Reshape to 2D grid
                        attn_2d = attn.reshape(patch_grid_size, patch_grid_size)
                        
                        # Normalize to 0-255 range
                        attn_normalized = (attn_2d - attn_2d.min()) / (attn_2d.max() - attn_2d.min() + 1e-8)
                        attn_uint8 = (attn_normalized * 255).astype(np.uint8)
                        
                        # Create PIL image and resize to 448x448

                        cmap = plt.get_cmap('jet')
                        attn_colored = cmap(attn_uint8 / 255.0)  # Normalize to [0,1]
                        attn_colored = (attn_colored[:, :, :3] * 255).astype(np.uint8)

                        attn_img = Image.fromarray(attn_colored)
                        attn_img_resized = attn_img.resize((448, 448), Image.BILINEAR)

                        # attn_img = Image.fromarray(attn_uint8, mode='L')
                        # attn_img_resized = attn_img.resize((448, 448), Image.BILINEAR)
                        
                        # Get prediction info
                        pred_class = torch.argmax(logits[b, task_idx]).item()
                        pred_prob = torch.softmax(logits[b, task_idx], dim=0)[pred_class].item()
                        
                        # Convert ground truth and prediction to labels
                        gt_value = gt_labels[task_idx]
                        if gt_value == 1.0:
                            gt_label = "positive"
                        elif gt_value == 0.0:
                            gt_label = "uncertain"
                        else:  # -1.0
                            gt_label = "negative"
                        
                        if gt_label == "uncertain":
                            # Skip saving uncertain samples to reduce clutter
                            continue

                        if pred_class == 0:
                            pred_label = "positive"
                        elif pred_class == 1:
                            pred_label = "uncertain"
                        else:  # 2
                            pred_label = "negative"
                        
                        # Create filename with disease name
                        # Clean disease name for filename (remove special characters)
                        clean_disease_name = disease_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
                        filename = f'{clean_disease_name}_gt_{gt_label}_pred_{pred_label}_prob_{pred_prob:.3f}.png'
                        
                        # Save attention map
                        save_path = os.path.join(sample_dir, filename)
                        attn_img_resized.save(save_path)
                    
                    saved_samples += 1
                    sample_idx += 1
            else:
                sample_idx += images.shape[0]
    
    
    logger.info(f"Saved attention maps for {saved_samples} test samples")







def train(model, train_loader, val_loader, optimizer, scheduler, args, rank, world_size, disease_names, class_weights_tensor=None):
    """Main training function with DDP support, early stopping, and class weights"""
    device = next(model.parameters()).device
    best_val_loss = float('inf')
    best_val_auc = 0.0
    start_epoch = 0
    
    # Get num_tasks from model
    if isinstance(model, DDP):
        num_tasks = model.module.num_tasks
    else:
        num_tasks = model.num_tasks
    
    # Resume from checkpoint if specified
    if args.resume:
        model, optimizer, scheduler, start_epoch, best_val_loss = load_checkpoint(
            model, optimizer, scheduler, args.resume, device)
    
    # Initialize early stopping (only on rank 0)
    early_stopping = None
    if args.early_stopping and rank == 0:
        mode = 'min' if args.early_stopping_metric == 'loss' else 'max'
        early_stopping = EarlyStopping(
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta,
            mode=mode,
            verbose=True
        )
        logger.info("=" * 80)
        logger.info(f"Loss Function selected = {args.loss_type}")
        logger.info("=" * 80)
        logger.info(f"Early stopping enabled:")
        logger.info(f"  Patience: {args.early_stopping_patience} epochs")
        logger.info(f"  Min delta: {args.early_stopping_min_delta}")
        logger.info(f"  Metric: {args.early_stopping_metric}")
        logger.info(f"  Mode: {mode}")
        if class_weights_tensor is not None:
            logger.info(f"  Using class weights: Yes")
        logger.info("=" * 80)
    
    logger.info(f"Rank {rank}: Starting training from epoch {start_epoch}")
    
    # Flag to signal early stopping across all ranks
    should_stop = torch.tensor([0], dtype=torch.int, device=device)
    
    for epoch in range(start_epoch, args.num_epochs):
        # Check if we should stop (broadcast from rank 0)
        if world_size > 1:
            dist.broadcast(should_stop, src=0)
        
        if should_stop.item() == 1:
            if rank == 0:
                logger.info(f"Early stopping triggered at epoch {epoch}")
            break
        
        # Set epoch for the distributed samplers
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        if isinstance(val_loader.sampler, DistributedSampler):
            val_loader.sampler.set_epoch(epoch)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_cls_loss = 0.0
        train_attn_loss = 0.0
        train_comp_loss = 0.0
        all_train_labels = []
        all_train_probas = {f'disease_{i}': [] for i in range(num_tasks)}
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train/Rank {rank}]", 
                          disable=rank != 0)
        
        for batch in train_bar:
            images = batch['image'].to(device)
            labels = batch['labels'].to(device)  # [batch, num_tasks] with values [1.0, 0.0, -1.0]
            
            # Store original labels for AUC calculation
            all_train_labels.append(labels.cpu().numpy())
            
            # Convert labels to [0, 1, 2] format for CrossEntropyLoss
            labels_converted = convert_labels_for_loss(labels)  # [batch, num_tasks]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass (without labels to get raw outputs)
            outputs = model(images)
            logits = outputs['logits']  # [batch, num_tasks, num_classes]
            

            if args.loss_type == 'ce':
                # Calculate weighted CE loss
                if class_weights_tensor is not None:
                    cls_loss = calculate_weighted_ce_loss(logits, labels_converted, class_weights_tensor)
                else:
                    # Fallback to unweighted loss
                    cls_loss = 0
                    for task_idx in range(num_tasks):
                        cls_loss += F.cross_entropy(logits[:, task_idx, :], labels_converted[:, task_idx])
                    cls_loss = cls_loss / num_tasks
            else:

                focal_loss_fn = FocalLoss(args.focal_alpha, args.focal_gamma)
                cls_loss = calculate_weighted_focal_loss(logits, labels_converted, focal_loss_fn, class_weights_tensor)
            
            # Calculate attention loss
            attentions = outputs['attentions']  # Now [batch, num_tasks, num_patches] tensor
            # attn_normalized = attentions / (attentions.sum(dim=-1, keepdim=True) + 1e-8)
            # entropy = -(attn_normalized * torch.log(attn_normalized + 1e-8)).sum(dim=-1)
            # attn_loss = entropy.mean()  # Already averaged across batch and tasks
            attn_relu = torch.relu(attentions)
            attn_loss = torch.sum(attn_relu)/(attn_relu.size(0) * attn_relu.size(1) * attn_relu.size(2) + 1e-8)
            attn_loss = torch.pow(attn_loss, 2)

            certain_mask = (labels != -1.0).float()
            
            # # Calculate the mean entropy *only* for certain (non-uncertain) labels
            # masked_entropy_sum = (entropy * certain_mask).sum()
            # num_certain = certain_mask.sum()
            
            # # Add epsilon to prevent division by zero if a batch has 0 certain labels
            # attn_loss = masked_entropy_sum / (num_certain + 1e-8)
            
            

            competition_loss = torch.tensor(0.0, device=device)
            if args.lambda_competition > 0 and num_tasks > 1:
                # attentions shape: [batch_size, num_tasks, num_patches]             
                attn_norm = F.normalize(attentions, p=2, dim=-1) # [B, T, P]
                
                # Calculate the pairwise cosine similarity matrix for each batch item
                # (B, T, P) @ (B, P, T) -> (B, T, T)
                # This matrix shows the similarity of every slot to every other slot
                cos_sim_matrix = torch.bmm(attn_norm, attn_norm.transpose(1, 2))
                
                # 1. Create the pairwise mask: [B, J, J]
                # M_ij = 1.0 if and only if both task i and task j are certain
                pairwise_mask = torch.bmm(
                certain_mask.unsqueeze(2),  # [B, J, 1]
                certain_mask.unsqueeze(1)   # [B, 1, J]
                )

                # Create a mask to zero out the diagonal (self-similarity)
                eye_mask = torch.eye(num_tasks, device=device).unsqueeze(0).bool()
                
                # Apply the mask. We use abs() because we want to penalize
                # both high positive (redundant) and high negative (anti-correlated) similarity.
                # We want them to be orthogonal (similarity = 0).

                # off_diagonal_sim = cos_sim_matrix.masked_fill(eye_mask, 0.0).abs()
                
                off_diagonal_sim = F.relu(cos_sim_matrix.masked_fill(eye_mask, 0.0).pow(3))
              
                masked_off_diagonal_sim = off_diagonal_sim * pairwise_mask
                # Calculate the mean similarity over all B*T*(T-1) pairs
                competition_loss = masked_off_diagonal_sim.sum() / (args.batch_size * num_tasks * (num_tasks - 1))



            # Total loss
            loss = (args.lambda_ce*cls_loss) + (args.lambda_attn_loss * attn_loss) + (args.lambda_competition * competition_loss)

            # Get probabilities for AUC (all classes needed for multi-class AUC)
            probas_dict = convert_logits_to_probas(logits, return_all_classes=True)
            for disease_key, probs in probas_dict.items():
                all_train_probas[disease_key].append(probs)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            # scheduler.step()

            # Update metrics
            train_loss += loss.item()
            train_cls_loss += cls_loss.item()
            train_attn_loss += attn_loss.item()
            train_comp_loss += competition_loss
            
            if rank == 0:
                train_bar.set_postfix({
                    'loss': loss.item(),
                    'cls': cls_loss.item(),
                    'attn': attn_loss.item(),
                    'comp': competition_loss
                })
        
        # Average training losses
        train_loss = torch.tensor(train_loss, device=device)
        train_cls_loss = torch.tensor(train_cls_loss, device=device)
        train_attn_loss = torch.tensor(train_attn_loss, device=device)
        train_comp_loss = torch.tensor(train_comp_loss, device=device)
        
        if world_size > 1:
            dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_cls_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_attn_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_comp_loss, op=dist.ReduceOp.SUM)
        
        avg_train_loss = (train_loss / len(train_loader)) / world_size
        avg_train_cls_loss = (train_cls_loss / len(train_loader)) / world_size
        avg_train_attn_loss = (train_attn_loss / len(train_loader)) / world_size
        avg_train_comp_loss = (train_comp_loss / len(train_loader)) / world_size

        # Calculate training AUCs on rank 0
        train_aucs = {}
        avg_train_auc = 0.0
        # --- NEW: GATHER RESULTS FROM ALL GPUs ---
        if world_size > 1:
            gathered_labels_lists = [None] * world_size
            gathered_probas_lists = [None] * world_size
            
            dist.all_gather_object(gathered_labels_lists, all_train_labels)
            dist.all_gather_object(gathered_probas_lists, all_train_probas)
            
            if rank == 0:
                # Re-assemble all_train_labels
                all_labels_complete = []
                for lst in gathered_labels_lists:
                    all_labels_complete.extend(lst)
                all_train_labels = np.concatenate(all_labels_complete, axis=0)

                # Re-assemble all_train_probas
                all_probas_complete = {f'disease_{i}': [] for i in range(num_tasks)}
                for proba_dict in gathered_probas_lists:
                    for i in range(num_tasks):
                        key = f'disease_{i}'
                        if key in proba_dict:
                           all_probas_complete[key].extend(proba_dict[key])
                
                all_train_probas_dict = {} 
                for i in range(num_tasks):
                    key = f'disease_{i}'
                    if all_probas_complete[key]:
                        all_train_probas_dict[key] = np.concatenate(all_probas_complete[key], axis=0)
                all_train_probas = all_train_probas_dict
                        
        elif rank == 0: # Non-DDP case
            all_train_labels = np.concatenate(all_train_labels, axis=0)
            for i in range(num_tasks):
                disease_key = f'disease_{i}'
                if disease_key in all_train_probas and len(all_train_probas[disease_key]) > 0:
                    all_train_probas[disease_key] = np.concatenate(all_train_probas[disease_key], axis=0)
        
        # --- END OF NEW GATHERING LOGIC ---

        # This part now runs *only* on rank 0, but with the *full* data
        if rank == 0:
            logger.info("Calculating Training AUCs on Rank 0 with gathered data...")
            train_aucs = calculate_aucs(all_train_labels, all_train_probas, disease_names)
            avg_train_auc = np.mean(list(train_aucs.values())) if train_aucs else 0.0
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_cls_loss = 0.0
        val_attn_loss = 0.0
        val_comp_loss = 0.0
        all_val_labels = []
        all_val_probas = {f'disease_{i}': [] for i in range(num_tasks)}
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Val/Rank {rank}]",
                           disable=rank != 0)
            
            for batch in val_bar:
                images = batch['image'].to(device)
                labels = batch['labels'].to(device)
                
                # Store labels for AUC calculation
                all_val_labels.append(labels.cpu().numpy())
                
                # Convert labels
                labels_converted = convert_labels_for_loss(labels)
                
                # Forward pass
                outputs = model(images)
                logits = outputs['logits']
                
                if args.loss_type == 'ce':
                    # Calculate weighted CE loss
                    if class_weights_tensor is not None:
                        cls_loss = calculate_weighted_ce_loss(logits, labels_converted, class_weights_tensor)
                    else:
                        cls_loss = 0
                        for task_idx in range(num_tasks):
                            cls_loss += F.cross_entropy(logits[:, task_idx, :], labels_converted[:, task_idx])
                        cls_loss = cls_loss / num_tasks
                else:
                    focal_loss_fn = FocalLoss(args.focal_alpha, args.focal_gamma)
                    cls_loss = calculate_weighted_focal_loss(logits, labels_converted, focal_loss_fn, class_weights_tensor)
                
                # Calculate attention loss
                attentions = outputs['attentions']  # Now [batch, num_tasks, num_patches] tensor
                # attn_normalized = attentions / (attentions.sum(dim=-1, keepdim=True) + 1e-8)
                # entropy = -(attn_normalized * torch.log(attn_normalized + 1e-8)).sum(dim=-1)
                # attn_loss = entropy.mean()  # Already averaged across batch and tasks
                attn_loss = attentions.sum()
                

                competition_loss = 0.0
                if args.lambda_competition > 0 and num_tasks > 1:
                    # attentions shape: [batch_size, num_tasks, num_patches]             
                    attn_norm = F.normalize(attentions, p=2, dim=-1) # [B, T, P]
                    cos_sim_matrix = torch.bmm(attn_norm, attn_norm.transpose(1, 2))
                    
                    
                    certain_mask = (labels != -1.0).float()
                    pairwise_mask = torch.bmm(
                        certain_mask.unsqueeze(2),  # [B, J, 1]
                        certain_mask.unsqueeze(1)   # [B, 1, J]
                        )
                    
                    # Create a mask to zero out the diagonal (self-similarity)
                    eye_mask = torch.eye(num_tasks, device=device).unsqueeze(0).bool()
                    # off_diagonal_sim = cos_sim_matrix.masked_fill(eye_mask, 0.0).abs()
                    off_diagonal_sim = F.relu(cos_sim_matrix.masked_fill(eye_mask, 0.0).pow(3))
                    # Calculate the mean similarity over all B*T*(T-1) pairs
                    masked_off_diagonal_sim = off_diagonal_sim * pairwise_mask
                    competition_loss = masked_off_diagonal_sim.sum() / (args.batch_size * num_tasks * (num_tasks - 1))


                # Total loss
                loss = (args.lambda_ce*cls_loss) + (args.lambda_attn_loss * attn_loss) + (args.lambda_competition * competition_loss)
                
                # Get probabilities for AUC
                probas_dict = convert_logits_to_probas(logits, return_all_classes=True)
                for disease_key, probs in probas_dict.items():
                    all_val_probas[disease_key].append(probs)
                
                # Update metrics
                val_loss += loss.item()
                val_cls_loss += cls_loss.item()
                val_attn_loss += attn_loss.item()
                val_comp_loss += competition_loss
                
                if rank == 0:
                    val_bar.set_postfix({
                        'loss': loss.item(),
                        'cls': cls_loss.item(),
                        'attn': attn_loss.item(),
                        'comp': competition_loss
                    })
        
        # Average validation losses
        val_loss = torch.tensor(val_loss, device=device)
        val_cls_loss = torch.tensor(val_cls_loss, device=device)
        val_attn_loss = torch.tensor(val_attn_loss, device=device)
        val_comp_loss = torch.tensor(val_comp_loss, device=device)
        
        if world_size > 1:
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_cls_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_attn_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_comp_loss, op=dist.ReduceOp.SUM)
        
        avg_val_loss = (val_loss / len(val_loader)) / world_size
        avg_val_cls_loss = (val_cls_loss / len(val_loader)) / world_size
        avg_val_attn_loss = (val_attn_loss / len(val_loader)) / world_size
        avg_val_comp_loss = (val_comp_loss / len(val_loader)) / world_size
        
        # Calculate validation AUCs on rank 0
        val_aucs = {}
        avg_val_auc = 0.0
        if rank == 0:
            all_val_labels = np.concatenate(all_val_labels, axis=0)
            for i in range(num_tasks):
                disease_key = f'disease_{i}'
                if disease_key in all_val_probas and len(all_val_probas[disease_key]) > 0:
                    all_val_probas[disease_key] = np.concatenate(all_val_probas[disease_key], axis=0)
            val_aucs = calculate_aucs(all_val_labels, all_val_probas, disease_names)
            avg_val_auc = np.mean(list(val_aucs.values())) if val_aucs else 0.0
        
        # Update LR scheduler
        scheduler.step()
        
        # Log metrics (only on rank 0)
        if rank == 0:
            logger.info(f"Epoch {epoch+1}/{args.num_epochs}")
            logger.info(f"Train - Loss: {avg_train_loss:.4f}, CE: {avg_train_cls_loss:.4f}, "
                       f"Attn: {avg_train_attn_loss:.4f}, Comp: {avg_train_comp_loss:.4f}, AUC: {avg_train_auc:.4f}")
            logger.info(f"Val   - Loss: {avg_val_loss:.4f}, CE: {avg_val_cls_loss:.4f}, "
                       f"Attn: {avg_val_attn_loss:.4f}, Comp: {avg_val_comp_loss:.4f}, AUC: {avg_val_auc:.4f}")
            
            # Log to WandB if enabled
            if args.need_wandb and WANDB_AVAILABLE:
                wandb_log = {
                    'epoch': epoch + 1,
                    'train/loss': avg_train_loss.item(),
                    'train/cls_loss': avg_train_cls_loss.item(),
                    'train/attn_loss': avg_train_attn_loss.item(),
                    'train/comp_loss': avg_train_comp_loss.item(),
                    'train/auc': avg_train_auc,
                    'val/loss': avg_val_loss.item(),
                    'val/cls_loss': avg_val_cls_loss.item(),
                    'val/attn_loss': avg_val_attn_loss.item(),
                    'val/comp_loss': avg_val_comp_loss.item(),
                    'val/auc': avg_val_auc,
                    'learning_rate': optimizer.param_groups[0]['lr']
                }
                wandb.log(wandb_log)
            
            # Save top 5 AUCs for training and validation
            top_train_aucs = {k: v for k, v in sorted(train_aucs.items(), key=lambda x: x[1], reverse=True)[:5]}
            top_val_aucs = {k: v for k, v in sorted(val_aucs.items(), key=lambda x: x[1], reverse=True)[:5]}
            
            logger.info(f"Top 5 Training AUCs: {top_train_aucs}")
            logger.info(f"Top 5 Validation AUCs: {top_val_aucs}")
        
        # Check early stopping (only on rank 0)
        if early_stopping is not None and rank == 0:
            if args.early_stopping_metric == 'loss':
                current_metric = avg_val_loss.item()
            else:
                current_metric = avg_val_auc
            
            if early_stopping(current_metric, epoch):
                should_stop[0] = 1
                logger.info("=" * 80)
                logger.info("EARLY STOPPING TRIGGERED")
                logger.info(f"Best {args.early_stopping_metric}: {early_stopping.best_score:.4f} at epoch {early_stopping.best_epoch + 1}")
                logger.info("=" * 80)
        
        # Save checkpoint only if it's the best (only on rank 0)
        is_best_auc = avg_val_auc > best_val_auc
        
        if is_best_auc:
            best_val_auc = avg_val_auc
            best_val_loss = avg_val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, args, is_best=True, rank=rank)
    
        if args.check_freq > 0 and (epoch + 1) % args.check_freq == 0:
            if rank == 0: # <-- ADD THIS CHECK
                logger.info(f"Rank 0: Triggering periodic save of attention maps...")
                periodic_save(args, val_loader, device, disease_names, num_tasks, epoch)
        
        # Add a barrier to make all processes wait until Rank 0 is done
        if world_size > 1:
            dist.barrier()

    if rank == 0:
        if early_stopping and early_stopping.early_stop:
            logger.info(f"Training stopped early at epoch {epoch+1}")
            logger.info(f"Best validation {args.early_stopping_metric} was {early_stopping.best_score:.4f} at epoch {early_stopping.best_epoch + 1}")
        else:
            logger.info("Training completed!")
    
    return model


def setup_process(rank, world_size, args):
    """Setup distributed training process"""
    
    # Set device based on availability and preference
    if args.use_gaudi and HABANA_AVAILABLE:
        device = torch.device("hpu")
        htcore.mark_step()
    else:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
    
    # Set up distributed backend
    if world_size > 1:
        backend = args.backend
        if args.use_gaudi and HABANA_AVAILABLE:
            backend = "hccl"
        
        dist.init_process_group(
            backend=backend,
            init_method=f"env://",
            world_size=world_size,
            rank=rank,
        )
    
    # Set up logging
    global logger
    logger = setup_logging(rank, args)
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize WandB (only on rank 0)
    if rank == 0 and args.need_wandb:
        if WANDB_AVAILABLE:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name if args.wandb_run_name else args.exp_name,
                config=vars(args)
            )
            logger.info("WandB initialized")
        else:
            logger.warning("WandB requested but not available. Install with: pip install wandb")
    
    logger.info(f"Rank {rank}: Using {device} device")
    if world_size > 1:
        logger.info(f"Rank {rank}/{world_size-1}: DDP initialized with backend '{backend}'")
    
    # Load dataset
    logger.info(f"Rank {rank}: Loading dataset...")
    
    dataframe = pd.read_csv(args.csv_file)
    dataframe = dataframe.dropna()
    

    train_indices, val_indices, test_indices = create_stratified_splits(
        dataframe,
        args.train_split,
        args.val_split,
        args.test_split,
        args.stratify_column,
        args.seed
    )

    train_df = dataframe.iloc[train_indices].reset_index(drop=True)
    val_df   = dataframe.iloc[val_indices].reset_index(drop=True)
    test_df  = dataframe.iloc[test_indices].reset_index(drop=True)

    # Now create datasets
    train_dataset = CXRDataset(
        df=train_df,
        root_dir=args.root_dir,
        transform=None,
        augment=True,
        augmentation_factor=1
    )

    val_dataset = CXRDataset(
        df=val_df,
        root_dir=args.root_dir,
        transform=None
    )

    test_dataset = CXRDataset(
        df=test_df,
        root_dir=args.root_dir,
        transform=None
    )
    
    # Get disease names for consistent ordering
    disease_names = train_dataset.label_cols
    num_tasks = len(disease_names)
    
    
    # Create processor and collate function
    processor = AutoProcessor.from_pretrained("google/medsiglip-448")
    collate_fn = create_collate_fn(processor)
    
    # Create distributed samplers
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=True,
            seed=args.seed
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=args.seed
        )
    else:
        train_sampler = None
        val_sampler = None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Create test loader (no distributed sampler needed for final evaluation)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    if rank == 0:
        logger.info(f"Total dataset size: {len(dataframe)}")
        logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        logger.info(f"Number of diseases/tasks: {num_tasks}")
        logger.info(f"Disease names: {disease_names[:5]}... (showing first 5)")
    
    # Calculate class weights from training data (only on rank 0, then broadcast)
    class_weights_tensor = None
    if args.use_class_weights:
        if rank == 0:
            class_weights_dict, class_weights_tensor = calculate_class_weights(
                train_loader, disease_names, device
            )
        else:
            # Create placeholder on other ranks
            class_weights_tensor = torch.zeros(num_tasks, 3, device=device)
        
        # Broadcast class weights to all ranks
        if world_size > 1:
            dist.broadcast(class_weights_tensor, src=0)
        
        if rank == 0:
            logger.info(f"Class weights broadcasted to all {world_size} ranks")
    else:
        if rank == 0:
            logger.info("Class weights disabled")
    
    # Create model
    model = MedSIGLIPSlotAttention(
        num_tasks=num_tasks,
        num_classes_per_task=3,
        hidden_dim=args.hidden_dim,
        slot_iters=args.slot_iters,
        freeze_backbone=args.freeze_backbone,
        lambda_attn_loss=args.lambda_attn_loss,
        reduced_dim=args.reduced_dim,
        disease_embeddings_path=args.embedding_path
    )
    model = model.to(device)
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
        logger.info(f"Reduced dimension: {args.reduced_dim}")
    
    # Wrap model with DDP if using multiple processes
    if world_size > 1:
        # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, find_unused_parameters=False)
        if rank == 0:
            logger.info("Model wrapped with DistributedDataParallel")
    
    # Define optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Define learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=2*args.num_epochs,
        eta_min=1e-8
    )

    # scheduler = CyclicLR(
    # optimizer,
    # base_lr=1e-7,       # minimum LR
    # max_lr=1e-4,         # maximum LR
    # step_size_up=1000,    # number of iterations to go from base_lr to max_lr
    # mode='triangular2'   # can also be 'triangular2' or 'exp_range'
    # )
    
    # Train the model
    if rank == 0:
        logger.info("=" * 80)
        logger.info("Starting training...")
        logger.info("=" * 80)
    
    train(model, train_loader, val_loader, optimizer, scheduler, args, rank, world_size, disease_names, class_weights_tensor)
    #############################################################################################################################################
    # Synchronize all processes before test evaluation
    if world_size > 1:
        dist.barrier()
    
    # Evaluate best checkpoint on test set (only on rank 0)
    if rank == 0:
        logger.info("\n" + "=" * 80)
        logger.info("Training completed! Starting test set evaluation...")
        logger.info("=" * 80 + "\n")
        
        # Unwrap DDP model for evaluation
        if isinstance(model, DDP):
            eval_model = model.module
        else:
            eval_model = model
        
        # Move model to evaluation device
        eval_model = eval_model.to(device)
        
        # Evaluate best checkpoint (with attention map saving for limited samples)
        results_df = evaluate_best_checkpoint(
            args, test_loader, device, disease_names, num_tasks,
            save_attention_maps=args.save_attention_maps
        )
        
        
        # Create summary visualization if results available
        if args.need_wandb and WANDB_AVAILABLE and results_df is not None:
            # Create a simple summary plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            metrics = ['avg_auc_3class', 'avg_f1_3class', 'avg_precision_3class', 'avg_recall_3class', 'avg_auc_binary', 'avg_f1_binary', 'avg_precision_binary', 'avg_recall_binary']
            values = [results_df['avg_auc_3class'].values[0], results_df['avg_f1_3class'].values[0], results_df['avg_precision_3class'].values[0], results_df['avg_recall_3class'].values[0], results_df['avg_auc_binary'].values[0], results_df['avg_f1_binary'].values[0], results_df['avg_precision_binary'].values[0], results_df['avg_recall_binary'].values[0]]
            
            bars = ax.bar(metrics, values, color=['#4CAF50', '#FF9800'], alpha=0.8, edgecolor='black', linewidth=2)
            ax.set_ylabel('Score', fontsize=14, fontweight='bold')
            ax.set_title('Test Set Performance - Best Model', fontsize=16, fontweight='bold')
            ax.set_ylim(0, 1.0)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.4f}',
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            summary_plot_path = os.path.join(args.checkpoint_dir, args.exp_name, "test_evaluation_summary.png")
            plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Summary plot saved to: {summary_plot_path}")
            
            # Log to WandB
            wandb.log({"test_evaluation_summary": wandb.Image(summary_plot_path)})
            plt.close()
    
    # Finish WandB run
    if rank == 0 and args.need_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    # Clean up
    if world_size > 1:
        dist.destroy_process_group()


def main():
    args = parse_args()
    
    # Get world size from environment variable (set by torchrun)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Set up logging and process
    setup_process(local_rank, world_size, args)


if __name__ == "__main__":
    main()