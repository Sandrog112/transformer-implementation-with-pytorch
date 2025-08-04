import torch
import torch.nn as nn

class PositionWiseFeedForward(nn.Module):
    """
    Position-wise feed forward neural network layer.

    Args:
        d_model (int): Input and output dimension
        d_ff (int): Hidden layer dimension
        
    Shape:
        - Input: (batch_size, seq_len, d_model)
        - Output: (batch_size, seq_len, d_model)
    """
    
    def __init__(self, d_model: int, d_ff: int):
        """
        Initializes the feed forward network layers.
        
        """
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feed forward network.
        
        Args:
            x (Tensor): Input tensor
            
        Returns:
            Tensor: Processed tensor through two linear layers with ReLU activation
        """
        return self.fc2(self.relu(self.fc1(x)))