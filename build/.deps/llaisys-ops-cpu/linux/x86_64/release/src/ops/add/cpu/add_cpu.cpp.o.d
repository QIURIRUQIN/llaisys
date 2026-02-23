{
    depfiles_format = "gcc",
    files = {
        "src/ops/add/cpu/add_cpu.cpp"
    },
    depfiles = "add_cpu.o: src/ops/add/cpu/add_cpu.cpp src/ops/add/cpu/add_cpu.hpp  include/llaisys.h src/ops/add/cpu/../../../utils.hpp  src/ops/add/cpu/../../../utils/check.hpp  src/ops/add/cpu/../../../utils/types.hpp\
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