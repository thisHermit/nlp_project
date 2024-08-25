import torch.nn as nn
import torch
import torch.nn.functional as F

# Focal Loss 
# implementation from https://pytorch.org/vision/main/_modules/torchvision/ops/focal_loss.html
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        Initializes the Focal Loss function.
        
        Args:
        gamma: Focusing parameter that reduces the relative loss for well-classified examples.
        alpha: A tensor of shape (C,) where C is the number of classes, used to balance the loss.
        reduction: Specifies the reduction to apply to the output ('none', 'mean', 'sum').
        """

        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Computes the Focal Loss between the predicted inputs and the true targets.
        
        Args:
        inputs: Predicted logits from the model of shape (N, C) where N is the batch size and C is the number of classes.
        targets: Ground truth labels of shape (N,) where each value is 0 ≤ targets[i] ≤ C-1.
        
        Returns: The computed Focal Loss.
        """

        BCE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt) ** self.gamma * BCE_loss

        if self.alpha is not None:
            alpha = torch.tensor(self.alpha).to(inputs.device)
            F_loss = alpha[targets] * F_loss

        # choosing the specified reduction
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss



# Dice Loss
# implementation from https://discuss.pytorch.org/t/implementation-of-dice-loss/53552
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        """
        Initializes the Dice Loss function.
        
        Args:
        smooth: A smoothing constant to prevent division by zero.
        """

        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Computes the Dice Loss between the predicted inputs and the true targets.
        
        Args:
        inputs: Predicted logits from the model of shape (N, C, H, W) where N is the batch size, C is the number of classes, H and W are the height and width.
        targets: Ground truth labels of shape (N, C, H, W).
        
        Returns: The computed Dice Loss.
        """

        inputs = F.softmax(inputs, dim=1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice
