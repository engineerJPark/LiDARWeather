from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import weight_reduce_loss
from mmengine.utils import is_list_of

from mmdet3d.registry import MODELS


@MODELS.register_module()
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, num_classes=19, ignore_index=19):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.num_classes = num_classes
        # self.ignore_index = ignore_index

    def forward(self, inputs, targets, ignore_index=19):
        """
        ignore_index=19 for semantickitti dataset
        """
        if ignore_index is not None:
            # Create mask for ignore index
            mask = (targets != ignore_index)
            # Apply mask to inputs and targets_one_hot
            inputs = inputs[mask]
            targets = targets[mask]

        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, self.num_classes).float()

        # Flatten the tensors
        inputs = inputs.reshape(-1)
        targets_one_hot = targets_one_hot.reshape(-1)
        
        # Calculate intersection and union
        intersection = (inputs * targets_one_hot).sum()
        union = inputs.sum() + targets_one_hot.sum()
        
        # Calculate Dice coefficient
        dice_coeff = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Dice loss is defined as 1 - Dice coefficient
        return 1 - dice_coeff

# Example usage:
if __name__ == "__main__":
    # Create some example tensors
    inputs = torch.tensor([[0.1, 0.4, 0.9], 
                           [0.7, 0.2, 0.2], 
                           [0.1, 0.9, 0.4]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    targets = torch.tensor([2, 1, 1], dtype=torch.int64).unsqueeze(0).unsqueeze(0)
    
    # Initialize Dice loss
    criterion = DiceLoss()
    
    # Calculate loss
    loss = criterion(inputs, targets)
    print(f'Dice Loss: {loss.item()}')