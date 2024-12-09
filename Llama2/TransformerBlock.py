import torch
import torch.nn as nn 
from RMSNorm import RMSNorm, FeedForward
from MultiHeadAttention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.attn = MultiHeadAttention(
            dim_in = cfg["emb_dim"],
            dim_out = cfg["emb_dim"],
            context_length= cfg["context_length"],
            num_heads= cfg["n_heads"],
            dtype=cfg["dtype"]
        )
        self.feedforward = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"])
        self.norm2 = RMSNorm(cfg["emb_dim"])
        #self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self,x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        #x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x 
        x = self.norm2(x)
        x = self.feedforward(x)
        #x = self.drop_shortcut(x)
        x = x + shortcut
        return x



