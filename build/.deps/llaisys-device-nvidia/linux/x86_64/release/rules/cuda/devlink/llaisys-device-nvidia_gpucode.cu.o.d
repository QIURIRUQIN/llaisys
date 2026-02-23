{
    values = {
        "/usr/local/cuda/bin/nvcc",
        {
            "-L/usr/local/cuda/lib64/stubs",
            "-L/usr/local/cuda/lib64",
            "-Lbuild/linux/x86_64/release",
            "-lcudart",
            "-lcuda",
            "-lllaisys-utils",
            "-lcudadevrt",
            "-lrt",
            "-lpthread",
            "-ldl",
            "-Xcompiler=-fPIC",
            "-m64",
            "-gencode",
            "arch=compute_89,code=sm_89",
            "-dlink"
        }
    },
    files = {
        "build/.objs/llaisys-device-nvidia/linux/x86_64/release/src/device/nvidia/nvidia_resource.cu.o",
        "build/.objs/llaisys-device-nvidia/linux/x86_64/release/src/device/nvidia/nvidia_runtime_api.cu.o",
        "build/linux/x86_64/release/libllaisys-utils.a"
    }
}