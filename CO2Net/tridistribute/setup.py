from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

if torch.cuda.is_available():
    print('Including CUDA code.')
    setup(
        name='tridistribute',
        ext_modules=[
            CUDAExtension('tridistribute', [
                'src/tridistribute_cuda.cpp',
                'src/tridistribute_kernel.cu',
            ])
        ],
        cmdclass={
            'build_ext': BuildExtension
        })
else:
    print('NO CUDA is found. Fall back to CPU.')
    setup(name='tridistribute',
        ext_modules=[CppExtension('tridistribute', ['src/tridistribute.cpp'])],
        cmdclass={'build_ext': BuildExtension})
