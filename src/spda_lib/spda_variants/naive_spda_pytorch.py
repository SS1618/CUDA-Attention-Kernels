import torch
from spda_lib.base import BaseSPDA
from spda_lib.registry import register_variants

@register_variants("naive_pytorch_spda")
class NaivePytorchSPDA(BaseSPDA):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V):
        return spda_kernel(Q, K, V)

def spda_kernel(Q, K, V):
    # Q, K, V: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = Q.shape
    output = torch.zeros_like(Q)  # (batch_size, num_heads, seq_len, head_dim)

    scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)  # (batch_size, num_heads, seq_len, seq_len)
    attn_weights = torch.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
    output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_len, head_dim)

    return output