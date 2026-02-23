{
    depfiles_format = "gcc",
    files = {
        "src/core/runtime/runtime.cpp"
    },
    depfiles = "runtime.o: src/core/runtime/runtime.cpp src/core/runtime/runtime.hpp  src/core/runtime/../core.hpp  src/core/runtime/../../device/runtime_api.hpp include/llaisys/runtime.h  include/llaisys/../llaisys.h src/core/runtime/../../device/../utils.hpp  src/core/runtime/../../device/../utils/check.hpp  src/core/runtime/../../device/../utils/types.hpp include/llaisys.h  src/core/runtime/../allocator/allocator.hpp  src/core/runtime/../allocator/../storage/storage.hpp  src/core/runtime/../allocator/naive_allocator.hpp\
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