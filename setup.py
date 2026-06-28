import os
import sys
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Resolve the absolute path to the directory containing setup.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Define compilation and optimization flags
# -O3 forces high host-compiler optimizations.
# --use_fast_math tells NVCC to trade minor precision for massive speed gains.
extra_compile_args = {
    'cxx': ['-O3'],
    'nvcc': ['-O3', '--use_fast_math', '-Xcompiler', '-fPIC']
}

# 2. Configure our single custom CUDA extension module
ext_modules = [
    CUDAExtension(
        # The 'attention_lib._C' layout forces the compiled binary file 
        # (_C.so or _C.pyd) to land straight inside your src/attention_lib/ directory.
        name='attention_lib._C',
        sources=[
            'kernels/binding.cpp',
            'kernels/cuda/naive_spda_kernel.cu',
        ],
        # Tells the preprocessor to use 'csrc/' as a base root path, 
        # allowing binding.cpp to safely call #include "naive_cuda/naive_spda_kernel.h"
        include_dirs=[
            os.path.join(BASE_DIR, 'kernels'),
        ],
        extra_compile_args=extra_compile_args
    )
]

# 3. Trigger the standard Setuptools execution pipeline
setup(
    name='attention_lib',
    version='0.1.0',
    description='A highly modular, pluggable attention benchmark harness',
    author='Saurav',
    
    # Directs python to treat the 'src/' folder as the root directory for packages
    package_dir={'': 'src'},
    
    # Automatically searches 'src/' and finds your 'attention_lib' package
    # along with all submodules like 'attention_lib.spda_variants'
    packages=find_packages(where='src'),
    
    # Mounts our custom compiled C++/CUDA extension module
    ext_modules=ext_modules,
    
    # Replaces the default build commands with PyTorch's specialized BuildExtension
    # script. This automatically hooks into local system installations of NVCC and Ninja.
    cmdclass={
        'build_ext': BuildExtension
    },
    
    # Core system requirements needed to run and evaluate this package
    install_requires=[
        'torch>=2.0.0',
        'triton>=2.0.0',
        'pyyaml',
    ],
    
    python_requires='>=3.8',
)
