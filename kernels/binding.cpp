#include <pybind11/pybind11.h>
#include "naive_sdpa_kernel.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("naive_sdpa_cuda", &naive_sdpa_cuda, "Naive SDPA kernel");
}