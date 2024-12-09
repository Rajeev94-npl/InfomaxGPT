import torch
import torch.nn as nn 

def precompute_rope_params(head_dim, theta_base=10_000, context_length=4096):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))

    # Generate position indices 
    positions = torch.arange(context_length)

    # Compute the angles 
    angles = positions[:,None] * inv_freq[None,:]    #Shape: (context_length, head_dim//2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)      #Shape: (context_length, head_dim)

    # Precompute sine and cosine 
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos,sin

def compute_rope(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half 
    x1 = x[...,: head_dim//2] #First half 
    x2 = x[..., head_dim//2:] #Second half 

    # Adjust sin and cos shapes 
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0) # Shape: (1,1,seq_len,head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation 
    rotated = torch.cat((-x2,x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_in, dim_out,
                 context_length, num_heads, dtype=None):
        super().__init__()
        assert (dim_out % num_heads == 0),\
                "dim_out must be divisible by num_heads"
        
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.head_dim = dim_out// num_heads
        self.W_query = nn.Linear(dim_in,dim_out,bias=False, dtype=dtype)
        self.W_key = nn.Linear(dim_in, dim_out,bias=False, dtype=dtype)
        self.W_value = nn.Linear(dim_in,dim_out,bias=False, dtype=dtype)
        self.out_heads = nn.Linear(dim_out,dim_out, bias=False, dtype=dtype)
        #self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length,context_length),diagonal=1)
        )

        cos,sin = precompute_rope_params(head_dim=self.head_dim, context_length=context_length)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        
    def forward(self, x):
        batch_size, num_tokens, dim_in = x.shape 
        
        keys = self.W_key(x) # Shape: (batch, num_tokens, dim_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        ) 

        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)

        #compute rope
        keys = compute_rope(keys, self.cos, self.sin)
        queries = compute_rope(queries, self.cos, self.sin)
        
        attention_scores = queries @ keys.transpose(2,3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attention_scores.masked_fill_(mask_bool, -torch.inf)

        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1]**0.5, dim=-1
        )
        #attention_weights = self.dropout(attention_weights)

        context_vector = (attention_weights @ values).transpose(1,2)

        context_vector = context_vector.reshape(batch_size, num_tokens, self.dim_out)
        context_vector = self.out_heads(context_vector)
        return context_vector
    

if __name__ == "__main__":
    # Settings
    batch_size = 2
    context_len = 5
    num_heads = 4
    head_dim = 16

    # Instantiate RoPE parameters
    cos, sin = precompute_rope_params(head_dim=head_dim, context_length=context_len)

    # Dummy query and key tensors
    torch.manual_seed(123)
    queries = torch.randn(batch_size, num_heads, context_len, head_dim)
    keys = torch.randn(batch_size, num_heads, context_len, head_dim)

    # Apply rotary position embeddings
    queries_rot = compute_rope(queries, cos, sin)
    keys_rot = compute_rope(keys, cos, sin)
    print("queries_rot",queries_rot)
    print("keys_rot",keys_rot)

    # Settings
    batch_size = 1
    context_len = 100
    max_context_len = 4096
    embed_dim = 128
    num_heads = 4


    example_batch = torch.randn((batch_size, context_len, embed_dim))

    mha = MultiHeadAttention(
        dim_in=embed_dim,
        dim_out=embed_dim,
        context_length=max_context_len,
        num_heads=num_heads
    )

    print(mha(example_batch).shape)

    del mha  # delete to free up memory