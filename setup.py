import os
import sys
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Resolve the absolute path to the directory containing setup.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Define compilation and optimization flags
extra_compile_args = {
    'cxx': ['-O3'],
    'nvcc': ['-O3', '--use_fast_math', '-Xcompiler', '-fPIC']
}

# 2. Configure our custom CUDA extension module mapped to sdpa_lib
ext_modules = [
    CUDAExtension(
        # This places the compiled binary directly into your real src/sdpa_lib/ directory
        name='sdpa_lib._C',
        sources=[
            'kernels/binding.cpp',
            'kernels/naive_cuda/naive_sdpa_kernel.cu',
        ],
        # Tells the preprocessor to use 'kernels/' as a base root path, 
        # allowing binding.cpp to safely call #include "naive_cuda/naive_sdpa_kernel.h"
        include_dirs=[
            os.path.join(BASE_DIR, 'kernels'),
            os.path.join(BASE_DIR, 'kernels/naive_cuda')
        ],
        extra_compile_args=extra_compile_args
    )
]

# 3. Trigger the standard Setuptools execution pipeline
setup(
    # Directs python to treat the 'src/' folder as the root directory for packages
    package_dir={'': 'src'},
    
    # Automatically searches 'src/' and finds your 'sdpa_lib' package
    packages=find_packages(where='src'),
    
    # Mounts our custom compiled C++/CUDA extension module
    ext_modules=ext_modules,
    
    # Replaces the default build commands with PyTorch's specialized BuildExtension script.
    cmdclass={
        'build_ext': BuildExtension
    },
)
