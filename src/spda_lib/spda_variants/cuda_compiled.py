import torch
import spda_lib._C as _C  # This accesses your single compiled C++ binary module
from spda_lib.base import BaseSPDA
from spda_lib.registry import register_variants

@register_variants("cuda_naive_spda")
class CudaNaiveSPDA(BaseSPDA):
    def forward(self, q, k, v):
        return _C.spda_kernel(q, k, v)