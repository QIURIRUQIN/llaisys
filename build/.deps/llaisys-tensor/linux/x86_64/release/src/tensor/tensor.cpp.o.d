{
    depfiles_format = "gcc",
    files = {
        "src/tensor/tensor.cpp"
    },
    depfiles = "tensor.o: src/tensor/tensor.cpp src/tensor/tensor.hpp  src/tensor/../core/llaisys_core.hpp src/tensor/../core/core.hpp  src/tensor/../core/context/context.hpp include/llaisys.h  src/tensor/../core/context/../runtime/runtime.hpp  src/tensor/../core/context/../runtime/../../device/runtime_api.hpp  include/llaisys/runtime.h include/llaisys/../llaisys.h  src/tensor/../core/context/../runtime/../../device/../utils.hpp  src/tensor/../core/context/../runtime/../../device/../utils/check.hpp  src/tensor/../core/context/../runtime/../../device/../utils/types.hpp  src/tensor/../core/context/../runtime/../allocator/allocator.hpp  src/tensor/../core/context/../runtime/../allocator/../storage/storage.hpp  src/tensor/../ops/rearrange/op.hpp\
",
    values = {
        "/usr/bin/g++",
        {
            "-m64",
            "-fvisibility=hidden",
            "-fvisibility-inlines-hidden",
            "-Wall",
            "-Werror",
            "-O3",
            "-std=c++17",
            "-Iinclude",
            "-DENABLE_NVIDIA_API",
            "-finput-charset=UTF-8",
            "-fexec-charset=UTF-8",
            "-fPIC",
            "-Wno-unknown-pragmas",
            "-DNDEBUG"
        }
    }
}