import torch
import torch.nn as nn 

def precompute_rope_params(head_dim, theta_base=10_000, context_length=4096, freq_config = None):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))

    # Frequency adjustments
    if freq_config is not None:
        low_freq_wavelen = freq_config["original_context_length"]/ freq_config["low_frequency_factor"]
        high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_frequency_factor"]
        
        wavelen = 2*torch.pi / inv_freq

        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen , inv_freq / freq_config["factor"],inv_freq
        )

        smooth_factor = (freq_config["original_context_length"]/wavelen - freq_config["low_freq_factor"])/(
            freq_config["high_freq_factor"] - freq_config["low_freq_factor"]
        )

        smoothed_inv_freq = (
            (1 - smooth_factor)*(inv_freq/freq_config["factor"]) + smooth_factor*inv_freq
        )

        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq,smoothed_inv_freq,inv_freq_llama)
        inv_freq = inv_freq_llama


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

class SharedBuffers:
    _buffers = {}

    @staticmethod
    def get_buffers(context_length, head_dim, rope_base, freq_config, dtype=torch.float32):
        key = (context_length, head_dim, rope_base, tuple(freq_config.values()) if freq_config else freq_config,dtype)

        if key not in SharedBuffers._buffers:
            #Create or fetch the buffers 
            mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
            cos, sin = precompute_rope_params(head_dim, rope_base, context_length, freq_config)
            if dtype is not None:
                cos = cos.to(dtype)
                sin = sin.to(dtype)
            SharedBuffers._buffers[key] = (mask, cos, sin)

        return SharedBuffers._buffers[key]
    


class GroupedQueryAttention(nn.Module):
    def __init__(self, dim_in, dim_out,
                 context_length, num_heads, num_kv_groups, rope_base=10_000,rope_config=None, dtype=None):
        super().__init__()
        assert (dim_out % num_heads == 0),\
                "dim_out must be divisible by num_heads"
        assert num_heads% num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"
        
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.head_dim = dim_out// num_heads
        
        self.W_key = nn.Linear(dim_in, num_kv_groups*self.head_dim,bias=False, dtype=dtype)
        self.W_value = nn.Linear(dim_in,num_kv_groups*self.head_dim,bias=False, dtype=dtype)
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups
        
        self.W_query = nn.Linear(dim_in,dim_out,bias=False, dtype=dtype)
        self.out_heads = nn.Linear(dim_out,dim_out, bias=False, dtype=dtype)
        
        #Fetch buffers using SharedBuffers 
        mask,cos, sin = SharedBuffers.get_buffers(context_length, self.head_dim, rope_base, rope_config, dtype)

        self.register_buffer("mask", mask)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        
    def forward(self, x):
        batch_size, num_tokens, dim_in = x.shape 
        
        keys = self.W_key(x) # Shape: (batch, num_tokens, dim_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim) 

        keys = keys.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim)
        

        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)

        #compute rope
        keys = compute_rope(keys, self.cos, self.sin)
        queries = compute_rope(queries, self.cos, self.sin)

        #Expand keys and values to match the number of heads
        # Shape: (batch_size, num_heads, num_tokens, head_dim)

        keys = keys.repeat_interleave(self.group_size, dim = 1) #Shape: (batch_size, num_heads, num_tokens, head_dim)
        values = values.repeat_interleave(self.group_size, dim=1) #Shape: (b, num_heads, num_tokens, head_dim)
        
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
    context_len = 8192
    num_heads = 4
    head_dim = 16

    # Instantiate RoPE parameters
    cos, sin = precompute_rope_params(head_dim=head_dim, theta_base=500_000, context_length=context_len)

    # Dummy query and key tensors
    torch.manual_seed(123)
    queries = torch.randn(batch_size, num_heads, context_len, head_dim)
    keys = torch.randn(batch_size, num_heads, context_len, head_dim)

    # Apply rotary position embeddings
    queries_rot = compute_rope(queries, cos, sin)
    keys_rot = compute_rope(keys, cos, sin)
    print("queries_rot",queries_rot.shape)
    print("keys_rot",keys_rot.shape)

    # Settings
    batch_size = 1
    context_len = 3000
    max_context_len = 8192
    embed_dim = 4096
    num_heads = 32


    example_batch = torch.randn((batch_size, context_len, embed_dim))

    gqa = GroupedQueryAttention(
        dim_in=embed_dim,
        dim_out=embed_dim,
        context_length=max_context_len,
        num_heads=num_heads,
        num_kv_groups=8,
        rope_base=500_000
    )

    print(gqa(example_batch).shape)
    print("W_key:", gqa.W_key.weight.shape)
    print("W_value:", gqa.W_value.weight.shape)
    print("W_query:", gqa.W_query.weight.shape)

    


    gqa_total_params = sum(p.numel() for p in gqa.parameters())
    print(f"GQA: {gqa_total_params:,}")

    del gqa  # delete to free up memory