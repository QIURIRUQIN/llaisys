{
    depfiles_format = "gcc",
    files = {
        "src/device/runtime_api.cpp"
    },
    depfiles = "runtime_api.o: src/device/runtime_api.cpp src/device/runtime_api.hpp  include/llaisys/runtime.h include/llaisys/../llaisys.h  src/device/../utils.hpp src/device/../utils/check.hpp  src/device/../utils/types.hpp include/llaisys.h\
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