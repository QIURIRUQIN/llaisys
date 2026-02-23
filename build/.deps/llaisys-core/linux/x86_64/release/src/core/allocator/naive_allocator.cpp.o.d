{
    depfiles_format = "gcc",
    files = {
        "src/core/allocator/naive_allocator.cpp"
    },
    depfiles = "naive_allocator.o: src/core/allocator/naive_allocator.cpp  src/core/allocator/naive_allocator.hpp src/core/allocator/allocator.hpp  include/llaisys/runtime.h include/llaisys/../llaisys.h  src/core/allocator/../storage/storage.hpp include/llaisys.h  src/core/allocator/../storage/../core.hpp  src/core/allocator/../runtime/runtime.hpp  src/core/allocator/../runtime/../../device/runtime_api.hpp  src/core/allocator/../runtime/../../device/../utils.hpp  src/core/allocator/../runtime/../../device/../utils/check.hpp  src/core/allocator/../runtime/../../device/../utils/types.hpp\
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