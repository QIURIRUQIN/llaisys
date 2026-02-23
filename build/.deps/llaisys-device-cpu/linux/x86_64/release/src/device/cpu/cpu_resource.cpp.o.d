{
    depfiles_format = "gcc",
    files = {
        "src/device/cpu/cpu_resource.cpp"
    },
    depfiles = "cpu_resource.o: src/device/cpu/cpu_resource.cpp  src/device/cpu/cpu_resource.hpp src/device/cpu/../device_resource.hpp  include/llaisys.h src/device/cpu/../../utils.hpp  src/device/cpu/../../utils/check.hpp  src/device/cpu/../../utils/types.hpp\
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