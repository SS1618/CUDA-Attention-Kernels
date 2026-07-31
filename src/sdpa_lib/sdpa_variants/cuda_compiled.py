import torch
import sdpa_lib._C as _C  # This accesses your single compiled C++ binary module
from sdpa_lib.base import BaseSDPA
from sdpa_lib.registry import register_variants

@register_variants("cuda_naive_sdpa")
class CudaNaiveSDPA(BaseSDPA):
    def forward(self, q, k, v):
        return _C.naive_sdpa_cuda(q, k, v)