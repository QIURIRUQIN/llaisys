{
    depfiles_format = "gcc",
    files = {
        "src/ops/linear/op.cpp"
    },
    depfiles = "op.o: src/ops/linear/op.cpp src/ops/linear/op.hpp  src/ops/linear/../../tensor/tensor.hpp  src/ops/linear/../../tensor/../core/llaisys_core.hpp  src/ops/linear/../../tensor/../core/core.hpp  src/ops/linear/../../tensor/../core/context/context.hpp  include/llaisys.h  src/ops/linear/../../tensor/../core/context/../runtime/runtime.hpp  src/ops/linear/../../tensor/../core/context/../runtime/../../device/runtime_api.hpp  include/llaisys/runtime.h include/llaisys/../llaisys.h  src/ops/linear/../../tensor/../core/context/../runtime/../../device/../utils.hpp  src/ops/linear/../../tensor/../core/context/../runtime/../../device/../utils/check.hpp  src/ops/linear/../../tensor/../core/context/../runtime/../../device/../utils/types.hpp  src/ops/linear/../../tensor/../core/context/../runtime/../allocator/allocator.hpp  src/ops/linear/../../tensor/../core/context/../runtime/../allocator/../storage/storage.hpp\
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