{
    depfiles_format = "gcc",
    files = {
        "src/llaisys/ops.cc"
    },
    depfiles = "ops.o: src/llaisys/ops.cc include/llaisys/ops.h include/llaisys/tensor.h  include/llaisys/../llaisys.h src/llaisys/llaisys_tensor.hpp  include/llaisys/tensor.h src/llaisys/../tensor/tensor.hpp  src/llaisys/../tensor/../core/llaisys_core.hpp  src/llaisys/../tensor/../core/core.hpp  src/llaisys/../tensor/../core/context/context.hpp include/llaisys.h  src/llaisys/../tensor/../core/context/../runtime/runtime.hpp  src/llaisys/../tensor/../core/context/../runtime/../../device/runtime_api.hpp  include/llaisys/runtime.h  src/llaisys/../tensor/../core/context/../runtime/../../device/../utils.hpp  src/llaisys/../tensor/../core/context/../runtime/../../device/../utils/check.hpp  src/llaisys/../tensor/../core/context/../runtime/../../device/../utils/types.hpp  src/llaisys/../tensor/../core/context/../runtime/../allocator/allocator.hpp  src/llaisys/../tensor/../core/context/../runtime/../allocator/../storage/storage.hpp  src/llaisys/../ops/add/op.hpp src/llaisys/../ops/argmax/op.hpp  src/llaisys/../ops/embedding/op.hpp src/llaisys/../ops/linear/op.hpp  src/llaisys/../ops/rearrange/op.hpp src/llaisys/../ops/rms_norm/op.hpp  src/llaisys/../ops/rope/op.hpp src/llaisys/../ops/self_attention/op.hpp  src/llaisys/../ops/swiglu/op.hpp\
",
    values = {
        "/usr/bin/g++",
        {
            "-m64",
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