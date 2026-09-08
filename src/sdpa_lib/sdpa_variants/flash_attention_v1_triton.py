import torch
from sdpa_lib.base import BaseSDPA
from sdpa_lib.registry import register_variants
import triton
import triton.language as tl

configs = [
    triton.Config({'BLOCK_R': 64,  'BLOCK_C': 32},  num_warps=4, num_stages=2),
    triton.Config({'BLOCK_R': 64,  'BLOCK_C': 64},  num_warps=4, num_stages=2),
    triton.Config({'BLOCK_R': 128, 'BLOCK_C': 64},  num_warps=4, num_stages=2),
    triton.Config({'BLOCK_R': 128, 'BLOCK_C': 64},  num_warps=8, num_stages=2),
    triton.Config({'BLOCK_R': 128, 'BLOCK_C': 128}, num_warps=8, num_stages=2),
]

def prune_invalid_configs(configs, named_args, **kwargs):
    d = kwargs['BLOCK_D']
    pruned = []
    for conf in configs:
        bm = conf.kwargs['BLOCK_R']
        bn = conf.kwargs['BLOCK_C']
        
        # Approximate FP16 shared memory requirement:
        # Q tile: (bm * d), K tile: (bn * d), V tile: (bn * d), O tile: (bm * d)
        # 2 bytes per element
        sram_bytes = 2 * (bm * d + 2 * bn * d + bm * d)
        
        # Ampere GA102 user block limit is ~96 KB (100 KB total minus driver headroom)
        if sram_bytes <= 96 * 1024:
            pruned.append(conf)
    return pruned

@register_variants("flash_attention_v1_triton")
class FlashAttentionV1Triton(BaseSDPA):
    def __init__(self):
            super().__init__()
    
    def forward(self, Q, K, V):
        return flash_attention_v1_triton(Q, K, V)

@triton.autotune(
    configs=configs,
    key=['SEQ_LEN', 'BLOCK_D'], # Retune when sequence length or head dim changes
    prune_configs_by={'early_config_prune': prune_invalid_configs}
)
@triton.jit
def flash_attention_v1_triton_kernel(Q_ptr, K_ptr, V_ptr, O_ptr, Sum_ptr, Maxes_ptr, SEQ_LEN: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    pid = tl.program_id(axis=0)



def flash_attention_v1_triton(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:

    assert Q.is_cuda and K.is_cuda and V.is_cuda

    batch_size, num_heads, seq_len, head_dim = Q.shape
    output = torch.empty_like(Q)
    sums = torch.empty((seq_len,), device=Q.device)
    maxes = torch.full((seq_len,), -float('inf'), device=Q.device)
    grid = (batch_size, num_heads)

    flash_attention_v1_triton_kernel[grid](Q, K, V, output, sums, maxes, seq_len, BLOCK_D = head_dim)

    return output