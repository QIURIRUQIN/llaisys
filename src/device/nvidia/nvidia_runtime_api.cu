#include "../runtime_api.hpp"

#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <optional>

namespace llaisys::device::nvidia {

namespace runtime_api {

// 简单的设备内存缓存池，减少频繁 cudaMalloc/cudaFree 开销。
// 按大小对齐存储，并限制池总量，避免占用过多显存。
namespace {
constexpr size_t kAlignBytes = 256;
constexpr size_t kMaxPoolBytes = 512ull << 20; // 512MB
constexpr size_t kPinnedThreshold = 1ull << 20; // 1MB 以上再做 pinned staging

inline size_t align_up(size_t n, size_t a) {
    return (n + a - 1) / a * a;
}

std::mutex pool_mu;
std::unordered_map<size_t, std::vector<void *>> free_blocks; // size -> blocks
std::unordered_map<void *, size_t> live_blocks;              // ptr -> size
size_t pool_bytes = 0;

// 主机端 pinned 缓冲池（用于 H2D / D2H staging）
std::unordered_map<size_t, std::vector<void *>> pinned_free_blocks;
std::unordered_map<void *, size_t> pinned_live_blocks;
size_t pinned_pool_bytes = 0;
constexpr size_t kPinnedMaxPoolBytes = 256ull << 20; // 256MB

bool is_pinned_host(const void *ptr) {
    cudaPointerAttributes attr;
    if (cudaPointerGetAttributes(&attr, ptr) != cudaSuccess) {
        cudaGetLastError(); // 清除可能的错误码
        return false;
    }
#if CUDART_VERSION >= 10000
    return attr.type == cudaMemoryTypeHost;
#else
    return attr.memoryType == cudaMemoryTypeHost;
#endif
}

void *alloc_pinned(size_t size) {
    const size_t aligned = align_up(size, kAlignBytes);
    {
        std::lock_guard<std::mutex> g(pool_mu);
        auto it = pinned_free_blocks.find(aligned);
        if (it != pinned_free_blocks.end() && !it->second.empty()) {
            void *p = it->second.back();
            it->second.pop_back();
            pinned_pool_bytes -= aligned;
            pinned_live_blocks[p] = aligned;
            return p;
        }
    }
    void *ptr = nullptr;
    cudaMallocHost(&ptr, aligned);
    std::lock_guard<std::mutex> g(pool_mu);
    pinned_live_blocks[ptr] = aligned;
    return ptr;
}

void release_pinned(void *ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> g(pool_mu);
    auto it = pinned_live_blocks.find(ptr);
    if (it == pinned_live_blocks.end()) {
        cudaFreeHost(ptr);
        return;
    }
    size_t sz = it->second;
    pinned_live_blocks.erase(it);
    if (pinned_pool_bytes + sz <= kPinnedMaxPoolBytes) {
        pinned_free_blocks[sz].push_back(ptr);
        pinned_pool_bytes += sz;
    } else {
        cudaFreeHost(ptr);
    }
}
} // namespace

// 获取可用的 NIVIDIA GPU 设备数量
int getDeviceCount() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return 0;
    }
    return count;
}

// 设置当前使用的 GPU 设备
void setDevice(int device_id) {
    cudaSetDevice(device_id);
}

// 同步设备，等待所有的 GPU 操作完成
void deviceSynchronize() {
    cudaDeviceSynchronize();
}

// 创建 CUDA 流，用于异步操作
llaisysStream_t createStream() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    return reinterpret_cast<llaisysStream_t>(stream);
}

// 销毁 CUDA 流
void destroyStream(llaisysStream_t stream) {
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cudaStreamDestroy(cuda_stream);
}

// 同步指定的 CUDA 流
void streamSynchronize(llaisysStream_t stream) {
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cudaStreamSynchronize(cuda_stream);
}

// 在 GPU 上分配设备内存
void *mallocDevice(size_t size) {
    const size_t aligned = align_up(size, kAlignBytes);

    { // 先尝试池子
        std::lock_guard<std::mutex> g(pool_mu);
        auto it = free_blocks.find(aligned);
        if (it != free_blocks.end() && !it->second.empty()) {
            void *ptr = it->second.back();
            it->second.pop_back();
            pool_bytes -= aligned;
            live_blocks[ptr] = aligned;
            return ptr;
        }
    }

    void *ptr = nullptr;
    cudaMalloc(&ptr, aligned);

    std::lock_guard<std::mutex> g(pool_mu);
    live_blocks[ptr] = aligned;
    return ptr;
}

// 释放设备 GPU 内存
void freeDevice(void *ptr) {
    if (ptr != nullptr) {
        std::lock_guard<std::mutex> g(pool_mu);
        auto it = live_blocks.find(ptr);
        if (it == live_blocks.end()) {
            cudaFree(ptr);
            return;
        }

        const size_t sz = it->second;
        live_blocks.erase(it);

        if (pool_bytes + sz <= kMaxPoolBytes) {
            free_blocks[sz].push_back(ptr);
            pool_bytes += sz;
        } else {
            cudaFree(ptr);
        }
    }
}
// 分配主机内存
void *mallocHost(size_t size) {
    void *ptr = nullptr;
    cudaMallocHost(&ptr, size);
    return ptr;
}

// 释放主机内存
void freeHost(void *ptr) {
    if (ptr != nullptr) {
        cudaFreeHost(ptr);
    }
}

// 同步内存拷贝
void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    cudaMemcpyKind cuda_kind;
    switch (kind) {
        case LLAISYS_MEMCPY_H2H:
            cuda_kind = cudaMemcpyHostToHost;
            break;
        case LLAISYS_MEMCPY_H2D:
            cuda_kind = cudaMemcpyHostToDevice;
            break;
        case LLAISYS_MEMCPY_D2H:
            cuda_kind = cudaMemcpyDeviceToHost;
            break;
        case LLAISYS_MEMCPY_D2D:
            cuda_kind = cudaMemcpyDeviceToDevice;
            break;
        default:
            cuda_kind = cudaMemcpyHostToHost;
            break;
    }

    const bool is_h2d = (cuda_kind == cudaMemcpyHostToDevice);
    const bool is_d2h = (cuda_kind == cudaMemcpyDeviceToHost);
    const bool large = size >= kPinnedThreshold;

    // 对大块 H2D/D2H 且未使用 pinned 内存时，采用 pinned staging 提升带宽
    if (large && (is_h2d || is_d2h)) {
        if (is_h2d && !is_pinned_host(src)) {
            void *pinned = alloc_pinned(size);
            std::memcpy(pinned, src, size);
            cudaMemcpy(dst, pinned, size, cudaMemcpyHostToDevice);
            release_pinned(pinned);
            return;
        } else if (is_d2h && !is_pinned_host(dst)) {
            void *pinned = alloc_pinned(size);
            cudaMemcpy(pinned, src, size, cudaMemcpyDeviceToHost);
            std::memcpy(dst, pinned, size);
            release_pinned(pinned);
            return;
        }
    }

    cudaMemcpy(dst, src, size, cuda_kind);
}

// 异步内存拷贝
void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cudaMemcpyKind cuda_kind;
    switch (kind) {
        case LLAISYS_MEMCPY_H2H:
            cuda_kind = cudaMemcpyHostToHost;
            break;
        case LLAISYS_MEMCPY_H2D:
            cuda_kind = cudaMemcpyHostToDevice;
            break;
        case LLAISYS_MEMCPY_D2H:
            cuda_kind = cudaMemcpyDeviceToHost;
            break;
        case LLAISYS_MEMCPY_D2D:
            cuda_kind = cudaMemcpyDeviceToDevice;
            break;
        default:
            cuda_kind = cudaMemcpyHostToHost;
            break;
    }
    cudaMemcpyAsync(dst, src, size, cuda_kind, cuda_stream);
}

// 
static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

// 强制引用 CUDA fatbin 注册符号，确保它们在链接时被包含
// 这是一个空的初始化函数，用于确保 CUDA 代码的初始化符号不被链接器丢弃
__attribute__((constructor))
static void init_cuda_fatbin() {
    // 这个函数会在共享库加载时自动调用
    // 确保 CUDA fatbin 注册符号被包含在最终的共享库中
}

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::nvidia
