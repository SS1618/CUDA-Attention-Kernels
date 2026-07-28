import sys
import torch
import pkgutil
import importlib

if not torch.cuda.is_available():
    sys.exit("CUDA is not available. Please ensure you have a compatible GPU and CUDA installed.")

from spda_lib.base import BaseSPDA
from spda_lib.registry import SPDA_REGISTRY
import spda_lib.spda_variants as spda_variants


for _, module_name, _ in pkgutil.iter_modules(spda_variants.__path__):
    importlib.import_module(f"spda_lib.spda_variants.{module_name}")

__version__ = "0.1.0"
__all__ = [
    "BaseSPDA",
    "SPDA_REGISTRY",
]
