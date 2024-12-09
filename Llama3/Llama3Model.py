import torch
import torch.nn as nn 
from RMSNorm import RMSNorm
from TransformerBlock import TransformerBlock

class Llama3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        #self.positional_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        #self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias= False, dtype=cfg["dtype"]
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        token_embeds = self.token_emb(in_idx)
        #position_embeds = self.positional_emb(torch.arange(seq_len, device=in_idx.device))
        x = token_embeds # + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        #x = self.drop_emb(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    

LLAMA3_CONFIG_8B = {
    "vocab_size": 128_256,   # Larger vocabulary size
    "context_length": 8192,  # Larger context length
    "emb_dim": 4096,         # Embedding dimension
    "n_heads": 32,           # Number of attention heads
    "n_layers": 32,          # Number of layers
    "hidden_dim": 14_336,    # Larger size of the intermediate dimension in FeedForward
    "n_kv_groups": 8,        # Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,  # The base in RoPE's "theta" was increased to 500_000
    "rope_freq": None,       # Additional configuration for adjusting the RoPE frequencies
    "dtype": torch.bfloat16  # Lower-precision dtype to reduce memory usage
}


if __name__ == "__main__":
    model = Llama3Model(LLAMA3_CONFIG_8B)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")


