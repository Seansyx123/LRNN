"""
model.py

LRNN model definition.

The model integrates a logic-aware LNN layer with a
neural classifier for software defect prediction.
"""

import torch
import torch.nn as nn

from .lnn_layer import LNNLayer


class LRNNModel(nn.Module):
    """
    Logic-Rule Neural Network (LRNN).

    Architecture:
        Input features
            -> LNNLayer (rule-aware feature adjustment)
            -> Fully-connected classifier
            -> Sigmoid output
    """

    def __init__(self, input_dim, penalty_weight=1.0):
        """
        Initialize LRNN model.

        Args:
            input_dim (int): Dimension of input feature vector.
            penalty_weight (float): Weight of rule penalty.
        """
        super().__init__()

        self.lnn_layer = LNNLayer(
            penalty_weight=penalty_weight
        )

        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, x, rules=None):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input feature tensor of shape (N, D).
            rules (list, optional): List of rule functions.

        Returns:
            torch.Tensor: Predicted defect probabilities.
        """
        # Convert tensor to numpy for rule evaluation
        if rules is not None:
            x_np = x.detach().cpu().numpy()
            x_np = self.lnn_layer.apply_penalty(
                x_np, rules
            )
            x = torch.tensor(
                x_np,
                dtype=x.dtype,
                device=x.device
            )

        logits = self.classifier(x)
        return torch.sigmoid(logits)
