import torch

# Assuming these class names exist inside your variants files
from sdpa_lib.sdpa_variants.naive_sdpa_pytorch import NaivePytorchSDPA
from sdpa_lib.sdpa_variants.cuda_compiled import CudaNaiveSDPA

device = torch.device("cuda")
print(device)
sdpa = NaivePytorchSDPA()

