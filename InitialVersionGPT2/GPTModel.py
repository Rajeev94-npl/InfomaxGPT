import torch
import torch.nn as nn 
from LayerNormalization import LayerNormalization
from TransformerBlock import TransformerBlock

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.positional_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNormalization(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias= False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        token_embeds = self.token_emb(in_idx)
        position_embeds = self.positional_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = token_embeds + position_embeds
        x = self.drop_emb(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    

GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Vocabulary Size
    "context_length": 1024,  # Context Length
    "emb_dim": 768,          # Embedding Dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Query-Key-Value bias
}

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:,-context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:,-1,:]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim= True)
        idx = torch.cat((idx,idx_next), dim=1)
        
        return idx


if __name__ == "__main__":
    import tiktoken

    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "You are a very"
    txt2 = "Ram is a very"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch,dim=0)
    #print(batch, batch.shape, batch.ndim)

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)

    out = model(batch)
    # print("Input batch:\n",batch, batch.shape)
    print("\nOutput shape:", out.shape)
    # print(out)
    total_params = sum(p.numel() for p in model.parameters())
    # print("Total number of parameters:", total_params)

    # print("Token embedding layer shape:", model.token_emb.weight.shape)
    # print("Output layer shape:", model.out_head.weight.shape)

    total_params_gpt2 = (total_params - sum(p.numel() for p in model.out_head.parameters()))
    #print(f"Number of trainable parameters considering weight tying:{total_params_gpt2:,}")

    #Finding total memory requirements 
    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes/ (1024*1024)
    #print(f"Total size of the model(in mb): {total_size_mb:.2f} mb")

    # # Testing the generate_text_simple function 

    start_context = "I am a"
    encoded = tokenizer.encode(start_context)
    print("encoded:",encoded)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("encoded tensor shape:", encoded_tensor.shape)

    model.eval()
    out = generate_text_simple(
        model=model,
        idx= encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    print("Output:", out)
    print("Output length:",len(out[0]))

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)



