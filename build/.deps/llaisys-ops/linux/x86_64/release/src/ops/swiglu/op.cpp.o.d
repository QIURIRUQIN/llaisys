{
    depfiles_format = "gcc",
    files = {
        "src/ops/swiglu/op.cpp"
    },
    depfiles = "op.o: src/ops/swiglu/op.cpp src/ops/swiglu/op.hpp  src/ops/swiglu/../../tensor/tensor.hpp  src/ops/swiglu/../../tensor/../core/llaisys_core.hpp  src/ops/swiglu/../../tensor/../core/core.hpp  src/ops/swiglu/../../tensor/../core/context/context.hpp  include/llaisys.h  src/ops/swiglu/../../tensor/../core/context/../runtime/runtime.hpp  src/ops/swiglu/../../tensor/../core/context/../runtime/../../device/runtime_api.hpp  include/llaisys/runtime.h include/llaisys/../llaisys.h  src/ops/swiglu/../../tensor/../core/context/../runtime/../../device/../utils.hpp  src/ops/swiglu/../../tensor/../core/context/../runtime/../../device/../utils/check.hpp  src/ops/swiglu/../../tensor/../core/context/../runtime/../../device/../utils/types.hpp  src/ops/swiglu/../../tensor/../core/context/../runtime/../allocator/allocator.hpp  src/ops/swiglu/../../tensor/../core/context/../runtime/../allocator/../storage/storage.hpp\
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