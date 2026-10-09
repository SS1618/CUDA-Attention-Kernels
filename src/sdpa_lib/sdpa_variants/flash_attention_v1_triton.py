import torch
from sdpa_lib.base import BaseSDPA
from sdpa_lib.registry import register_variants
import triton
import triton.language as tl

configs = [
    triton.Config({'BLOCK_R': 32, 'BLOCK_C': 32}, num_warps=4),
    triton.Config({'BLOCK_R': 32, 'BLOCK_C': 64}, num_warps=4),
    triton.Config({'BLOCK_R': 64, 'BLOCK_C': 64}, num_warps=4),
    triton.Config({'BLOCK_R': 64, 'BLOCK_C': 128}, num_warps=8),
    triton.Config({'BLOCK_R': 128, 'BLOCK_C': 128}, num_warps=8),
]

@register_variants("flash_attention_v1_triton")
class FlashAttentionV1Triton(BaseSDPA):
    def __init__(self):
            super().__init__()
    
    def forward(self, Q, K, V):
        return flash_attention_v1_triton(Q, K, V)

@triton.autotune(
    configs=configs,
    key=['SEQ_LEN', 'HEAD_DIM'], # Retune when sequence length or head dim changes
    )
@triton.jit
def flash_attention_v1_triton_kernel(Q_ptr, K_ptr, V_ptr, O_ptr, 
    Sum_ptr, Maxes_ptr, SEQ_LEN: tl.constexpr, HEAD_DIM: tl.constexpr, HEAD_SIZE: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):

    batch_id = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)
    Tr = (SEQ_LEN + BLOCK_R - 1) // BLOCK_R
    Tc = (SEQ_LEN + BLOCK_C - 1) // BLOCK_C

    k_block_ptr = tl.make_block_ptr(
        base=K_ptr + (batch_id * HEAD_SIZE * SEQ_LEN * HEAD_DIM) + (head_id * SEQ_LEN * HEAD_DIM),
        shape=(HEAD_DIM, SEQ_LEN),
        strides=(1, HEAD_DIM),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_C),
        order=(0, 1) #transposing
    )
    v_block_ptr = tl.make_block_ptr(
        base=V_ptr + (batch_id * HEAD_SIZE * SEQ_LEN * HEAD_DIM) + (head_id * SEQ_LEN * HEAD_DIM),
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(HEAD_DIM, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_C, HEAD_DIM),
        order=(1, 0)
    )
    
    summaxes_offsets = (batch_id * HEAD_SIZE * SEQ_LEN) + (head_id * SEQ_LEN) + tl.arange(0, BLOCK_R);

    for j in tl.range(0, Tc):
        K_j = tl.load(k_block_ptr, boundary_check=(1,))
        V_j = tl.load(v_block_ptr, boundary_check=(0,))
        q_block_ptr = tl.make_block_ptr(
            base=Q_ptr + (batch_id * HEAD_SIZE * SEQ_LEN * HEAD_DIM) + (head_id * SEQ_LEN * HEAD_DIM),
            shape=(SEQ_LEN, HEAD_DIM),
            strides=(HEAD_DIM, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_R, HEAD_DIM),
            order=(1, 0)
        )
        o_block_ptr = tl.make_block_ptr(
            base=O_ptr + (batch_id * HEAD_SIZE * SEQ_LEN * HEAD_DIM) + (head_id * SEQ_LEN * HEAD_DIM),
            shape=(SEQ_LEN, HEAD_DIM),
            strides=(HEAD_DIM, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_R, HEAD_DIM),
            order=(1, 0)
        )
        for i in tl.range(0, Tr):
            Q_i = tl.load(q_block_ptr, boundary_check=(0,))
            O_i = tl.load(o_block_ptr, boundary_check=(0,))
            flat_mask = (i * BLOCK_R) + summaxes_offsets < (batch_id * HEAD_SIZE * SEQ_LEN) + (head_id * SEQ_LEN) + SEQ_LEN
            Sum_i = tl.load(Sum_ptr + (i * BLOCK_R) + summaxes_offsets, mask=flat_mask, other=0.0)
            Maxes_i = tl.load(Maxes_ptr + (i * BLOCK_R) + summaxes_offsets, mask = flat_mask, other=-float('inf'))

            S_ij = tl.dot(Q_i, K_j, input_precision="ieee") * (1/ (HEAD_DIM ** 0.5))

            local_col_idx = (j * BLOCK_C) + tl.arange(0, BLOCK_C)
            col_mask = local_col_idx < SEQ_LEN

            S_ij = tl.where(col_mask[None, :], S_ij, -float('inf'))

            m_ij = tl.max(S_ij, axis=1)
            P_ij = tl.exp(S_ij - m_ij[:, None])
            l_ij = tl.sum(P_ij, axis=1)

            m_i_new = tl.maximum(Maxes_i, m_ij)
            l_i_new = tl.exp(Maxes_i - m_i_new) * Sum_i + tl.exp(m_ij - m_i_new) * l_ij
            O_i_new = tl.where(l_i_new > 0.0, 1.0 / l_i_new, 0.0)[:, None] * ((Sum_i[:, None] * tl.exp(Maxes_i - m_i_new)[:, None] * O_i) + (tl.exp(m_ij - m_i_new)[:, None] * tl.dot(P_ij, V_j)))

            tl.store(o_block_ptr, O_i_new, boundary_check=(0,))
            tl.store(Sum_ptr + (i * BLOCK_R) + summaxes_offsets, l_i_new, mask = flat_mask)
            tl.store(Maxes_ptr + (i * BLOCK_R) + summaxes_offsets, m_i_new, mask = flat_mask)

            q_block_ptr = tl.advance(q_block_ptr, (BLOCK_R, 0))
            o_block_ptr = tl.advance(o_block_ptr, (BLOCK_R, 0))

        k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_C))
        v_block_ptr = tl.advance(v_block_ptr, (BLOCK_C, 0))


def flash_attention_v1_triton(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:

    assert Q.is_cuda and K.is_cuda and V.is_cuda

    batch_size, num_heads, seq_len, head_dim = Q.shape
    output = torch.zeros_like(Q)
    sums = torch.zeros(batch_size, num_heads, seq_len).cuda()
    maxes = torch.full((batch_size, num_heads,seq_len), -torch.inf).cuda()
    grid = (batch_size, num_heads)

    flash_attention_v1_triton_kernel[grid](Q, K, V, output, sums, maxes, SEQ_LEN = seq_len, HEAD_DIM = head_dim, HEAD_SIZE = num_heads)

    return output