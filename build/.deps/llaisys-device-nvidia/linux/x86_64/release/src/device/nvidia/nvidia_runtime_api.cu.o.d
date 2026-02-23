{
    depfiles_format = "gcc",
    files = {
        "src/device/nvidia/nvidia_runtime_api.cu"
    },
    depfiles = "build/.objs/llaisys-device-nvidia/linux/x86_64/release/src/device/nvidia/nvidia_runtime_api.cu.o : src/device/nvidia/nvidia_runtime_api.cu     /usr/local/cuda/include/cuda_runtime.h     /usr/local/cuda/include/crt/host_config.h     /usr/local/cuda/include/builtin_types.h     /usr/local/cuda/include/device_types.h     /usr/local/cuda/include/crt/host_defines.h     /usr/local/cuda/include/driver_types.h     /usr/local/cuda/include/vector_types.h     /usr/local/cuda/include/surface_types.h     /usr/local/cuda/include/texture_types.h     /usr/local/cuda/include/library_types.h     /usr/local/cuda/include/channel_descriptor.h     /usr/local/cuda/include/cuda_runtime_api.h     /usr/local/cuda/include/cuda_device_runtime_api.h     /usr/local/cuda/include/driver_functions.h     /usr/local/cuda/include/vector_functions.h     /usr/local/cuda/include/vector_functions.hpp     /usr/local/cuda/include/crt/common_functions.h     /usr/local/cuda/include/crt/math_functions.h     /usr/local/cuda/include/crt/math_functions.hpp     /usr/local/cuda/include/crt/device_functions.h     /usr/local/cuda/include/crt/device_functions.hpp     /usr/local/cuda/include/device_atomic_functions.h     /usr/local/cuda/include/device_atomic_functions.hpp     /usr/local/cuda/include/crt/device_double_functions.h     /usr/local/cuda/include/crt/device_double_functions.hpp     /usr/local/cuda/include/sm_20_atomic_functions.h     /usr/local/cuda/include/sm_20_atomic_functions.hpp     /usr/local/cuda/include/sm_32_atomic_functions.h     /usr/local/cuda/include/sm_32_atomic_functions.hpp     /usr/local/cuda/include/sm_35_atomic_functions.h     /usr/local/cuda/include/sm_60_atomic_functions.h     /usr/local/cuda/include/sm_60_atomic_functions.hpp     /usr/local/cuda/include/sm_20_intrinsics.h     /usr/local/cuda/include/sm_20_intrinsics.hpp     /usr/local/cuda/include/sm_30_intrinsics.h     /usr/local/cuda/include/sm_30_intrinsics.hpp     /usr/local/cuda/include/sm_32_intrinsics.h     /usr/local/cuda/include/sm_32_intrinsics.hpp     /usr/local/cuda/include/sm_35_intrinsics.h     /usr/local/cuda/include/sm_61_intrinsics.h     /usr/local/cuda/include/sm_61_intrinsics.hpp     /usr/local/cuda/include/crt/sm_70_rt.h     /usr/local/cuda/include/crt/sm_70_rt.hpp     /usr/local/cuda/include/crt/sm_80_rt.h     /usr/local/cuda/include/crt/sm_80_rt.hpp     /usr/local/cuda/include/crt/sm_90_rt.h     /usr/local/cuda/include/crt/sm_90_rt.hpp     /usr/local/cuda/include/crt/sm_100_rt.h     /usr/local/cuda/include/crt/sm_100_rt.hpp     /usr/local/cuda/include/texture_indirect_functions.h     /usr/local/cuda/include/surface_indirect_functions.h     /usr/local/cuda/include/crt/cudacc_ext.h     /usr/local/cuda/include/device_launch_parameters.h     src/device/nvidia/../runtime_api.hpp     include/llaisys/runtime.h     include/llaisys/../llaisys.h     src/device/nvidia/../../utils.hpp     src/device/nvidia/../../utils/check.hpp     src/device/nvidia/../../utils/types.hpp     include/llaisys.h\
",
    values = {
        "/usr/local/cuda/bin/nvcc",
        {
            "-Xcompiler",
            "-Wall",
            "-Werror",
            "cross-execution-space-call,reorder,deprecated-declarations",
            "-Xcompiler",
            "-Werror",
            "-O3",
            "-Iinclude",
            "-I/usr/local/cuda/include",
            "--std",
            "c++17",
            "-DENABLE_NVIDIA_API",
            "-Xcompiler=-Wall",
            "-Xcompiler=-Werror",
            "-Xcompiler=-fPIC",
            "--extended-lambda",
            "--expt-relaxed-constexpr",
            "-Xcompiler=-Wno-error=deprecated-declarations",
            "-Xcompiler=-Wno-unknown-pragmas",
            "-m64",
            "-rdc=true",
            "-gencode",
            "arch=compute_89,code=sm_89",
            "-DNDEBUG"
        }
    }
}