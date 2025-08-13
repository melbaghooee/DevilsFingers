import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SeesawLoss(nn.Module):
    """
    Seesaw Loss for Long-Tailed Classification
    
    Paper: "Seesaw Loss for Long-Tailed Instance Segmentation"
    
    Args:
        num_classes: Number of classes
        p: Mitigation factor for rare classes (default: 0.8)
        q: Compensation factor for frequent classes (default: 2.0)
        eps: Small constant to avoid division by zero (default: 1e-2)
    """
    def __init__(self, num_classes, p=0.8, q=2.0, eps=1e-2):
        super(SeesawLoss, self).__init__()
        self.num_classes = num_classes
        self.p = p
        self.q = q
        self.eps = eps
        
        # Initialize cumulative sample counts for each class
        self.register_buffer('cumulative_samples', torch.zeros(num_classes))
        
    def forward(self, logits, targets):
        """
        Args:
            logits: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
        """
        batch_size = logits.size(0)
        
        # Update cumulative sample counts
        for i in range(batch_size):
            self.cumulative_samples[targets[i]] += 1
        
        # Calculate mitigation factor (reduces penalty for rare classes)
        mitigation_factor = torch.pow(self.cumulative_samples + self.eps, -self.p)
        mitigation_factor = mitigation_factor / mitigation_factor.max()
        
        # Calculate compensation factor (increases penalty for frequent classes)
        compensation_factor = torch.pow(self.cumulative_samples + self.eps, self.q)
        compensation_factor = compensation_factor / compensation_factor.max()
        
        # Apply softmax to logits
        probs = F.softmax(logits, dim=1)
        
        # Calculate seesaw weights for each sample
        seesaw_weights = torch.zeros_like(probs)
        for i in range(batch_size):
            target_class = targets[i]
            
            # For the target class: apply mitigation
            seesaw_weights[i, target_class] = mitigation_factor[target_class]
            
            # For non-target classes: apply compensation
            mask = torch.ones(self.num_classes, dtype=torch.bool, device=logits.device)
            mask[target_class] = False
            seesaw_weights[i, mask] = compensation_factor[mask]
        
        # Calculate weighted cross-entropy loss
        log_probs = F.log_softmax(logits, dim=1)
        weighted_log_probs = log_probs * seesaw_weights
        
        # Gather log probabilities for target classes
        target_log_probs = weighted_log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Return negative log likelihood
        loss = -target_log_probs.mean()
        
        return loss

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    Paper: "Focal Loss for Dense Object Detection"
    
    Args:
        alpha: Weighting factor [0, 1] (default: 1.0)
        gamma: Focusing parameter (default: 2.0)
        reduction: Reduction method ('mean', 'sum', 'none')
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, logits, targets):
        """
        Args:
            logits: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
