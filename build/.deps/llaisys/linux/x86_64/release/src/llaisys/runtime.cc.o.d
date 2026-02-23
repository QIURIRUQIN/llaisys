{
    depfiles_format = "gcc",
    files = {
        "src/llaisys/runtime.cc"
    },
    depfiles = "runtime.o: src/llaisys/runtime.cc include/llaisys/runtime.h  include/llaisys/../llaisys.h src/llaisys/../core/context/context.hpp  include/llaisys.h src/llaisys/../core/context/../core.hpp  src/llaisys/../core/context/../runtime/runtime.hpp  src/llaisys/../core/context/../runtime/../../device/runtime_api.hpp  src/llaisys/../core/context/../runtime/../../device/../utils.hpp  src/llaisys/../core/context/../runtime/../../device/../utils/check.hpp  src/llaisys/../core/context/../runtime/../../device/../utils/types.hpp  src/llaisys/../core/context/../runtime/../allocator/allocator.hpp  src/llaisys/../core/context/../runtime/../allocator/../storage/storage.hpp\
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