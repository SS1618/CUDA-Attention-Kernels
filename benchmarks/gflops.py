import torch

# Assuming these class names exist inside your variants files
from spda_lib.spda_variants.naive_spda_pytorch import NaivePytorchSPDA
from spda_lib.spda_variants.cuda_compiled import CudaNaiveSPDA

device = torch.device("cuda")
print(device)
spda = NaivePytorchSPDA()

