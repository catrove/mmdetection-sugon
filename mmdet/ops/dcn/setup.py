from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

setup(
    name="my_dcn_cuda",
    ext_modules=[
        CUDAExtension("deform_conv_cuda",['src/deform_conv_cuda_kernel.cu', 'src/deform_conv_cuda.cpp'],extra_compile_args={
            'hipcc':['-fno-gpu-rdc'],
            'cxx' : [],
            }),
        CUDAExtension("deform_pool_cuda",['src/deform_pool_cuda_kernel.cu', 'src/deform_pool_cuda.cpp'], extra_compile_args={
            'hipcc':['-fno-gpu-rdc'],
            'cxx' : [],
})
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
