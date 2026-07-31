import torch
import torch.nn as nn
import kernels


class NaiveAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(NaiveAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"
        
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        batch_size = query.size(0)
        
        # Linear projections
        query = self.query_linear(query).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        key = self.key_linear(key).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        value = self.value_linear(value).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        attn_output = kernels.naive_sdpa_pytorch.sdpa_kernel(query, key, value)

        # Final linear layer
        output = self.out_linear(attn_output)
        
        return output