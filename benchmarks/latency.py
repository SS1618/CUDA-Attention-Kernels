import torch
import torch.nn.functional as F
from sdpa_lib.registry import create_sdpa_variant, SDPA_REGISTRY
import argparse

def measure_gpu_latency(q, k, v, candidate_sdpa, num_warmup=10, num_iters=100):
    # Warm-up iterations
    for _ in range(num_warmup):
        _ = candidate_sdpa.forward(q, k, v)
    
    torch.cuda.synchronize()  # Ensure all warm-up operations are complete

    # Measure latency
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        _ = candidate_sdpa.forward(q, k, v)
    end_event.record()

    torch.cuda.synchronize()  # Wait for all operations to finish
    elapsed_time_ms = start_event.elapsed_time(end_event) / num_iters  # Average time per iteration
    return elapsed_time_ms

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure GPU latency of variant")
    
    # Take the name as a required CLI argument
    parser.add_argument(
        "variant_name", 
        type=str, 
        choices=SDPA_REGISTRY.keys(),  # Restricts input to valid registered keys
        help="The name of the variant to run."
    )
    
    args = parser.parse_args()

    query = torch.randn(2, 8, 1024, 64, dtype=torch.float32, device="cuda")
    key   = torch.randn(2, 8, 1024, 64, dtype=torch.float32, device="cuda")
    value = torch.randn(2, 8, 1024, 64, dtype=torch.float32, device="cuda")

    latency_ms = measure_gpu_latency(query, key, value, create_sdpa_variant(args.variant_name))
    print(f"Average latency for {args.variant_name}: {latency_ms:.4f} ms")