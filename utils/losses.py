# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class FocalLoss(nn.Module):
#     """
#     Focal Loss for dealing with class imbalance and hard examples.
    
#     Args:
#         alpha (float): Weighting factor for the rare class (typically the positive class)
#         gamma (float): Focusing parameter that reduces the loss contribution from easy examples
#         reduction (str): 'mean', 'sum' or 'none'
#     """
#     def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
        
#     def forward(self, inputs, targets):
#         # BCE loss
#         BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
#         # Convert logits to probabilities
#         pt = torch.exp(-BCE_loss)
        
#         # Compute focal weight
#         focal_weight = (1 - pt) ** self.gamma
        
#         # Apply alpha weighting
#         if self.alpha is not None:
#             focal_weight = self.alpha * focal_weight * targets + (1 - self.alpha) * focal_weight * (1 - targets)
            
#         # Compute final loss
#         loss = focal_weight * BCE_loss
        
#         # Apply reduction
#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         else:  # 'none'
#             return loss
            
# class CombinedLoss(nn.Module):
#     """
#     Combines BCE loss with Focal Loss for better performance.
    
#     Args:
#         focal_weight (float): Weight for focal loss contribution
#         bce_weight (float): Weight for BCE loss contribution
#         alpha (float): Alpha parameter for focal loss
#         gamma (float): Gamma parameter for focal loss
#     """
#     def __init__(self, focal_weight=0.5, bce_weight=0.5, alpha=0.25, gamma=2.0):
#         super(CombinedLoss, self).__init__()
#         self.focal_weight = focal_weight
#         self.bce_weight = bce_weight
#         self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
#         self.bce_loss = nn.BCEWithLogitsLoss()
        
#     def forward(self, inputs, targets):
#         focal = self.focal_loss(inputs, targets)
#         bce = self.bce_loss(inputs, targets)
#         return self.focal_weight * focal + self.bce_weight * bce