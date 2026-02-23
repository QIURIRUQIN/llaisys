#include "tensor.hpp"

#include "../utils.hpp"
#include "../ops/rearrange/op.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

// 根据形状计算步长
// 分配设备或主机内存
tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

// 获取张量数据指针
std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

// 获取维度
size_t Tensor::ndim() const {
    return _meta.shape.size();
}

// 获取形状
const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

// 获取步长
const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

// 获取数据类型
llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

// 获取设备信息
llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

// 计算元素总数
size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

// 获取单个元素大小
size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    if (_meta.shape.empty()) {
        return true;
    }
    
    // 检查步长是否符合连续布局
    // 连续布局：从最后一个维度开始，步长应该是 1, shape[-1], shape[-1]*shape[-2], ...
    size_t expected_stride = 1;
    for (size_t i = _meta.shape.size(); i > 0; i--) {
        size_t dim_idx = i - 1;
        if (_meta.strides[dim_idx] != static_cast<ptrdiff_t>(expected_stride)) {
            return false;
        }
        expected_stride *= _meta.shape[dim_idx];
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    CHECK_ARGUMENT(order.size() == _meta.shape.size(), "Permute order size must match tensor dimensions");
    
    // 验证 order 是有效的排列
    std::vector<bool> used(_meta.shape.size(), false);
    for (size_t i = 0; i < order.size(); i++) {
        CHECK_ARGUMENT(order[i] < _meta.shape.size(), "Permute order index out of range");
        CHECK_ARGUMENT(!used[order[i]], "Permute order contains duplicate indices");
        used[order[i]] = true;
    }
    
    // 创建新的形状和步长
    std::vector<size_t> new_shape(_meta.shape.size());
    std::vector<ptrdiff_t> new_strides(_meta.strides.size());
    
    for (size_t i = 0; i < order.size(); i++) {
        new_shape[i] = _meta.shape[order[i]];
        new_strides[i] = _meta.strides[order[i]];
    }
    
    TensorMeta new_meta{_meta.dtype, new_shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    // 计算新形状的元素总数
    size_t new_numel = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    size_t old_numel = this->numel();
    CHECK_ARGUMENT(new_numel == old_numel, "View shape must have the same number of elements");
    
    // 检查张量是否连续，如果不连续则不能创建视图
    CHECK_ARGUMENT(this->isContiguous(), "View requires contiguous tensor");
    
    // 计算新形状的步长
    size_t ndim = shape.size();
    std::vector<ptrdiff_t> new_strides(ndim);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim; i++) {
        new_strides[ndim - i] = stride;
        stride *= shape[ndim - i];
    }
    
    TensorMeta new_meta{_meta.dtype, shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    CHECK_ARGUMENT(dim < _meta.shape.size(), "Slice dimension out of range");
    CHECK_ARGUMENT(start <= end, "Slice start must be <= end");
    CHECK_ARGUMENT(end <= _meta.shape[dim], "Slice end out of range");
    
    // 创建新的形状和步长
    std::vector<size_t> new_shape = _meta.shape;
    std::vector<ptrdiff_t> new_strides = _meta.strides;
    
    new_shape[dim] = end - start;
    
    // 计算新的偏移量：offset += start * stride[dim]
    size_t new_offset = _offset + start * _meta.strides[dim] * this->elementSize();
    
    TensorMeta new_meta{_meta.dtype, new_shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, new_offset));
}

void Tensor::load(const void *src_) {
    core::context().setDevice(this->deviceType(), this->deviceId());
    size_t size_bytes = this->numel() * this->elementSize();
    
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        // CPU 设备：直接内存拷贝
        std::memcpy(this->data(), src_, size_bytes);
    } else {
        // GPU 设备：从主机到设备的内存拷贝
        core::context().runtime().api()->memcpy_sync(
            this->data(),
            src_,
            size_bytes,
            LLAISYS_MEMCPY_H2D
        );
    }
}

tensor_t Tensor::contiguous() const {
    // 如果已经是连续的，直接返回新的共享指针（共享 storage，不复制数据）
    if (this->isContiguous()) {
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    }
    
    // 创建新的连续张量
    auto new_tensor = create(_meta.shape, _meta.dtype, this->deviceType(), this->deviceId());
    
    // 创建一个临时的非 const Tensor 对象用于 rearrange
    // 注意：这里创建一个新的 Tensor 对象，但它共享相同的 storage
    auto temp_tensor = std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    
    core::context().setDevice(this->deviceType(), this->deviceId());
    llaisys::ops::rearrange(new_tensor, temp_tensor);
    
    return new_tensor;
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    // reshape 首先尝试使用 view，如果失败则先 contiguous 再 view
    size_t new_numel = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    size_t old_numel = this->numel();
    CHECK_ARGUMENT(new_numel == old_numel, "Reshape shape must have the same number of elements");
    
    // 如果张量是连续的，可以直接使用 view
    if (this->isContiguous()) {
        return this->view(shape);
    }
    
    // 否则，先创建连续副本，再 reshape
    return this->contiguous()->view(shape);
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    // 如果已经在目标设备上，直接返回新的共享指针
    if (this->deviceType() == device_type && 
        (device < 0 || this->deviceId() == device)) {
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    }
    
    // 确定目标设备 ID
    int target_device = (device >= 0) ? device : 0;
    
    // 创建目标设备上的新张量
    auto new_tensor = create(_meta.shape, _meta.dtype, device_type, target_device);
    
    // 确定内存拷贝类型并设置正确的设备上下文
    llaisysMemcpyKind_t copy_kind;
    if (this->deviceType() == LLAISYS_DEVICE_CPU && device_type != LLAISYS_DEVICE_CPU) {
        // H2D: 设置目标设备（GPU）
        copy_kind = LLAISYS_MEMCPY_H2D;
        core::context().setDevice(device_type, target_device);
    } else if (this->deviceType() != LLAISYS_DEVICE_CPU && device_type == LLAISYS_DEVICE_CPU) {
        // D2H: 设置源设备（GPU）
        copy_kind = LLAISYS_MEMCPY_D2H;
        core::context().setDevice(this->deviceType(), this->deviceId());
    } else {
        // D2D: 设置目标设备（GPU）
        copy_kind = LLAISYS_MEMCPY_D2D;
        core::context().setDevice(device_type, target_device);
    }
    
    size_t size_bytes = this->numel() * this->elementSize();
    core::context().runtime().api()->memcpy_sync(
        new_tensor->data(),
        this->data(),
        size_bytes,
        copy_kind
    );
    
    return new_tensor;
}

} // namespace llaisys
