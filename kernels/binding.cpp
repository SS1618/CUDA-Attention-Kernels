#include <pybind11/pybind11.h>
#include "naive_spda_kernel.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("naive_spda_cuda", &naive_spda_cuda, "Naive SPDA kernel");
}