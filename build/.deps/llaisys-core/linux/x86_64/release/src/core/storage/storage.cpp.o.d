{
    depfiles_format = "gcc",
    files = {
        "src/core/storage/storage.cpp"
    },
    depfiles = "storage.o: src/core/storage/storage.cpp src/core/storage/storage.hpp  include/llaisys.h src/core/storage/../core.hpp  src/core/storage/../runtime/runtime.hpp  src/core/storage/../runtime/../../device/runtime_api.hpp  include/llaisys/runtime.h include/llaisys/../llaisys.h  src/core/storage/../runtime/../../device/../utils.hpp  src/core/storage/../runtime/../../device/../utils/check.hpp  src/core/storage/../runtime/../../device/../utils/types.hpp  src/core/storage/../runtime/../allocator/allocator.hpp\
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