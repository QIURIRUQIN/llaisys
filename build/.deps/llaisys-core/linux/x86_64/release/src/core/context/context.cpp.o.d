{
    depfiles_format = "gcc",
    files = {
        "src/core/context/context.cpp"
    },
    depfiles = "context.o: src/core/context/context.cpp src/core/context/context.hpp  include/llaisys.h src/core/context/../core.hpp  src/core/context/../runtime/runtime.hpp  src/core/context/../runtime/../../device/runtime_api.hpp  include/llaisys/runtime.h include/llaisys/../llaisys.h  src/core/context/../runtime/../../device/../utils.hpp  src/core/context/../runtime/../../device/../utils/check.hpp  src/core/context/../runtime/../../device/../utils/types.hpp  src/core/context/../runtime/../allocator/allocator.hpp  src/core/context/../runtime/../allocator/../storage/storage.hpp\
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