import sys
import torch

if not torch.cuda.is_available():
    sys.exit("CUDA is not available. Please ensure you have a compatible GPU and CUDA installed.")

from spda_lib.base import BaseSPDA
from spda_lib.registry import SPDA_REGISTRY

from spda_lib import spda_variants
__version__ = "0.1.0"
__all__ = [
    "BaseSPDA",
    "SPDA_REGISTRY",
    "spda_variants",
]
