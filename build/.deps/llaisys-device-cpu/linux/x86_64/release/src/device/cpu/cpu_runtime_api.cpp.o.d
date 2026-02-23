{
    depfiles_format = "gcc",
    files = {
        "src/device/cpu/cpu_runtime_api.cpp"
    },
    depfiles = "cpu_runtime_api.o: src/device/cpu/cpu_runtime_api.cpp  src/device/cpu/../runtime_api.hpp include/llaisys/runtime.h  include/llaisys/../llaisys.h src/device/cpu/../../utils.hpp  src/device/cpu/../../utils/check.hpp  src/device/cpu/../../utils/types.hpp include/llaisys.h\
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