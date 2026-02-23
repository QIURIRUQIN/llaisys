{
    depfiles_format = "gcc",
    files = {
        "src/ops/add/op.cpp"
    },
    depfiles = "op.o: src/ops/add/op.cpp src/ops/add/op.hpp  src/ops/add/../../tensor/tensor.hpp  src/ops/add/../../tensor/../core/llaisys_core.hpp  src/ops/add/../../tensor/../core/core.hpp  src/ops/add/../../tensor/../core/context/context.hpp include/llaisys.h  src/ops/add/../../tensor/../core/context/../runtime/runtime.hpp  src/ops/add/../../tensor/../core/context/../runtime/../../device/runtime_api.hpp  include/llaisys/runtime.h include/llaisys/../llaisys.h  src/ops/add/../../tensor/../core/context/../runtime/../../device/../utils.hpp  src/ops/add/../../tensor/../core/context/../runtime/../../device/../utils/check.hpp  src/ops/add/../../tensor/../core/context/../runtime/../../device/../utils/types.hpp  src/ops/add/../../tensor/../core/context/../runtime/../allocator/allocator.hpp  src/ops/add/../../tensor/../core/context/../runtime/../allocator/../storage/storage.hpp  src/ops/add/cpu/add_cpu.hpp\
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