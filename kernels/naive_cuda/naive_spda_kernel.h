#pragma once
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

at::Tensor naive_spda_cuda(
    at::Tensor query,
    at::Tensor key,
    at::Tensor value
);