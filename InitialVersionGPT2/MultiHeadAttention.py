import torch
import torch.nn as nn 

class CausalAttention(nn.Module):
    def __init__(self, dim_in, dim_out, context_length,
                 dropout, qkv_bias = False):
        super().__init__()
        self.dim_out = dim_out
        self.W_query = nn.Linear(dim_in, dim_out, bias= qkv_bias)
        self.W_key = nn.Linear(dim_in, dim_out, bias= qkv_bias)
        self.W_value = nn.Linear(dim_in, dim_out, bias= qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        batch, num_tokens, dim_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attention_scores = queries @ keys.transpose(1,2)
        attention_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1]**0.5, dim=-1
        )
        attention_weights = self.dropout(attention_weights)

        context_vector = attention_weights @ values
        return context_vector


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, dim_in, dim_out, context_length, 
                 dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(
                dim_in, dim_out, context_length, dropout, qkv_bias
            ) for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_in, dim_out,
                 context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (dim_out % num_heads == 0),\
                "dim_out must be divisible by num_heads"
        
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.head_dim = dim_out// num_heads
        self.W_query = nn.Linear(dim_in,dim_out,bias=qkv_bias)
        self.W_key = nn.Linear(dim_in, dim_out,bias=qkv_bias)
        self.W_value = nn.Linear(dim_in,dim_out,bias=qkv_bias)
        self.out_heads = nn.Linear(dim_out,dim_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length,context_length),diagonal=1)
        )

    def forward(self, x):
        batch_size, num_tokens, dim_in = x.shape 
        keys = self.W_key(x)
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

        attention_scores = queries @ keys.transpose(2,3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attention_scores.masked_fill_(mask_bool, -torch.inf)

        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1]**0.5, dim=-1
        )
        attention_weights = self.dropout(attention_weights)

        context_vector = (attention_weights @ values).transpose(1,2)

        context_vector = context_vector.contiguous().view(
            batch_size, num_tokens, self.dim_out
        )
        context_vector = self.out_heads(context_vector)
        return context_vector
    
    
