import torch
import torch.nn as nn 
from RMSNorm import RMSNorm
from TransformerBlock import TransformerBlock

class Llama2Model(nn.Module):
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
    

LLAMA2_CONFIG_7B = {
    "vocab_size": 32000,     # Vocabulary Size
    "context_length": 4096,  # Context Length
    "emb_dim": 4096,          # Embedding Dimension
    "n_heads": 32,           # Number of attention heads
    "n_layers": 32,          # Number of layers
    "hidden_dim": 11008,     # Size of the intermediate dimension in FeedForward
    "dtype": torch.bfloat16  # Lower-precision dtype to reduce memory usage 
}


if __name__ == "__main__":
    model = Llama2Model(LLAMA2_CONFIG_7B)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")


