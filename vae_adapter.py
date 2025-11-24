import torch
import torch.nn as nn

class Feature_Adapter(nn.Module):
    def __init__(self, channel, alpha = 0.1):
        super().__init__()
        self.alpha = 0.1 
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(),
            nn.Linear(channel, channel)
        ) 

    def forward(self, x):
        # x has shape -> B, C, D, H, W
        B, C, D, H, W = x.shape
        # permute: B, D, H, W, C
        x_flattened = x.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        # Output
        y = self.mlp(x_flattened)
        y = y.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()

        return self.alpha * y + (1 - self.alpha) * x