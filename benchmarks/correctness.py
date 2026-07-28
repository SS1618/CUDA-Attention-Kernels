import torch
import torch.nn.functional as F
from spda_lib.registry import create_spda_variant, SPDA_REGISTRY
import argparse

def check_corrrectness(q, k, v, candidate_spda):
    reference_output = F.scaled_dot_product_attention(q, k, v)
    candidate_output = candidate_spda.forward(q, k, v)

    assert reference_output.shape == candidate_output.shape, f"shape mismatch {reference_output.shape} vs {candidate_output.shape}"
    nan_mismatch = torch.isnan(candidate_output) ^ torch.isnan(reference_output)
    inf_mismatch = torch.isinf(candidate_output) ^ torch.isinf(reference_output)
    if nan_mismatch.any() or inf_mismatch.any():
        print(f"NaN/Inf disagreement: {nan_mismatch.sum().item()} NaN, {inf_mismatch.sum().item()} Inf")
    close = torch.isclose(reference_output, candidate_output, rtol=1e-4, atol=1e-4)
    print(f"{(~close).sum().item()} / {reference_output.numel()} elements exceed tolerance "
          f"({100*(~close).sum().item()/reference_output.numel():.3f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check correctness of variant")
    
    # Take the name as a required CLI argument

    print(SPDA_REGISTRY.keys())

    parser.add_argument(
        "variant_name", 
        type=str, 
        choices=SPDA_REGISTRY.keys(),  # Restricts input to valid registered keys
        help="The name of the variant to run."
    )
    
    args = parser.parse_args()

    query = torch.randn(2, 8, 1024, 64, dtype=torch.float32, device="cuda")
    key   = torch.randn(2, 8, 1024, 64, dtype=torch.float32, device="cuda")
    value = torch.randn(2, 8, 1024, 64, dtype=torch.float32, device="cuda")

    check_corrrectness(query, key, value, create_spda_variant(args.variant_name))
    
