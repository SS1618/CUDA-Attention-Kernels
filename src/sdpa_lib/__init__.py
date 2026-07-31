import sys
import torch
import pkgutil
import importlib

if not torch.cuda.is_available():
    sys.exit("CUDA is not available. Please ensure you have a compatible GPU and CUDA installed.")

from sdpa_lib.base import BaseSDPA
from sdpa_lib.registry import SDPA_REGISTRY
import sdpa_lib.sdpa_variants as sdpa_variants


for _, module_name, _ in pkgutil.iter_modules(sdpa_variants.__path__):
    importlib.import_module(f"sdpa_lib.sdpa_variants.{module_name}")

__version__ = "0.1.0"
__all__ = [
    "BaseSDPA",
    "SDPA_REGISTRY",
]
