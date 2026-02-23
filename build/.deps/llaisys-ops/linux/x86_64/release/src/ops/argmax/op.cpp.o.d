{
    depfiles_format = "gcc",
    files = {
        "src/ops/argmax/op.cpp"
    },
    depfiles = "op.o: src/ops/argmax/op.cpp src/ops/argmax/op.hpp  src/ops/argmax/../../tensor/tensor.hpp  src/ops/argmax/../../tensor/../core/llaisys_core.hpp  src/ops/argmax/../../tensor/../core/core.hpp  src/ops/argmax/../../tensor/../core/context/context.hpp  include/llaisys.h  src/ops/argmax/../../tensor/../core/context/../runtime/runtime.hpp  src/ops/argmax/../../tensor/../core/context/../runtime/../../device/runtime_api.hpp  include/llaisys/runtime.h include/llaisys/../llaisys.h  src/ops/argmax/../../tensor/../core/context/../runtime/../../device/../utils.hpp  src/ops/argmax/../../tensor/../core/context/../runtime/../../device/../utils/check.hpp  src/ops/argmax/../../tensor/../core/context/../runtime/../../device/../utils/types.hpp  src/ops/argmax/../../tensor/../core/context/../runtime/../allocator/allocator.hpp  src/ops/argmax/../../tensor/../core/context/../runtime/../allocator/../storage/storage.hpp\
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