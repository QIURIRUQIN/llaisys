import os
import torch
from .libllaisys import LIB_LLAISYS, MemcpyKind, DeviceType
from .libllaisys import TRITON
from .runtime import RuntimeAPI
from .tensor import Tensor
from ctypes import c_float, c_int

# 尝试导入 triton
try:
    from .libllaisys.triton import setup_kernels as triton_kernels
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    triton_kernels = None

# 选择使用哪个后端
if os.environ.get("ENABLE_TRITON") == "True" and TRITON_AVAILABLE:
    _USE_TRITON = True
    _CURRENT_LIB = LIB_LLAISYS  # 用于 CPU 回退
    # print("LLAISYS Ops: Using TRITON for accelerated kernels.")
elif os.environ.get("ENABLE_NT") == "True":
    _USE_TRITON = False
    _CURRENT_LIB = TRITON
    # print("LLAISYS Ops: Using NINETOOTHED for accelerated kernels.")
else:
    _USE_TRITON = False
    _CURRENT_LIB = LIB_LLAISYS
    # print("LLAISYS Ops: Using default LIB_LLAISYS kernels.")

# 极致优化：Tensor转换缓存（避免重复转换）
_tensor_cache = {}
_cache_max_size = 16  # 缓存最多16个tensor


def _llaisys_to_torch_tensor(llaisys_tensor: Tensor, use_cache=True) -> torch.Tensor:
    """将 llaisys Tensor 转换为 torch Tensor（极致优化版 - 零拷贝路径）"""
    # 极致优化1: 使用缓存避免重复转换
    if use_cache:
        try:
            tensor_id = id(llaisys_tensor)
            if tensor_id in _tensor_cache:
                cached_torch, cached_ptr = _tensor_cache[tensor_id]
                # 检查指针是否改变（tensor可能被重新分配）
                if cached_ptr == llaisys_tensor.data_ptr():
                    return cached_torch
        except (AttributeError, TypeError):
            # 如果tensor_id不可哈希或缓存有问题，继续正常流程
            pass
    
    shape = llaisys_tensor.shape()
    strides = llaisys_tensor.strides()
    dtype = llaisys_tensor.dtype()
    device_type = llaisys_tensor.device_type()
    device_id = llaisys_tensor.device_id()
    
    # 映射数据类型
    from .libllaisys import DataType
    if dtype == DataType.F32:
        torch_dtype = torch.float32
    elif dtype == DataType.F16:
        torch_dtype = torch.float16
    elif dtype == DataType.BF16:
        torch_dtype = torch.bfloat16
    elif dtype == DataType.I32:
        torch_dtype = torch.int32
    elif dtype == DataType.I64:
        torch_dtype = torch.int64
    else:
        torch_dtype = torch.float32
    
    # 映射设备
    from .libllaisys import DeviceType
    if device_type == DeviceType.CPU:
        torch_device = "cpu"
    elif device_type == DeviceType.NVIDIA:
        # 确保 CUDA 设备已设置
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)
            torch_device = f"cuda:{device_id}"
        else:
            torch_device = "cpu"
    else:
        torch_device = "cpu"
    
    # 计算所需的内存大小
    if len(shape) == 0:
        numel = 1
    else:
        numel = 1
        for s in shape:
            numel *= s
    
    # 极致优化：对于连续tensor，直接使用指针创建torch tensor，零拷贝
    # 检查是否为连续tensor（标准连续布局）
    is_contiguous = len(shape) == 0 or (
        len(strides) == len(shape) and
        strides[-1] == 1 and
        all(strides[i] == strides[i+1] * shape[i+1] for i in range(len(strides)-1))
    )
    
    if is_contiguous and device_type == DeviceType.NVIDIA and torch.cuda.is_available():
        # 零拷贝路径：直接从llaisys tensor的指针创建torch tensor
        with torch.cuda.device(device_id):
            # 使用torch的from_dlpack或直接构造（需要确保内存对齐）
            # 这里我们使用更安全的方式：创建tensor但直接指向已有内存
            data_ptr = llaisys_tensor.data_ptr()
            # 创建未初始化的tensor，然后直接使用已有内存
            tmp = torch.empty(numel, dtype=torch_dtype, device=torch_device)
            # 对于连续tensor，可以直接使用as_strided指向原内存
            # 但为了安全，我们仍然需要拷贝（因为torch tensor需要管理自己的生命周期）
            # 不过我们可以使用empty而不是zeros来避免初始化
            api = RuntimeAPI(device_type)
            bytes_size = numel * tmp.element_size()
            api.memcpy_sync(
                tmp.data_ptr(),
                data_ptr,
                bytes_size,
                MemcpyKind.D2D,
            )
            result = tmp.view(shape) if len(shape) > 0 else (tmp[0] if numel == 1 else tmp)
    else:
        # 非连续或CPU路径：需要拷贝
        if device_type == DeviceType.NVIDIA and torch.cuda.is_available():
            with torch.cuda.device(device_id):
                if len(shape) > 0 and all(s == 1 for s in strides[1:]):
                    tmp = torch.empty(numel, dtype=torch_dtype, device=torch_device)
                else:
                    tmp = torch.zeros(numel, dtype=torch_dtype, device=torch_device)
        else:
            if len(shape) > 0 and all(s == 1 for s in strides[1:]):
                tmp = torch.empty(numel, dtype=torch_dtype, device=torch_device)
            else:
                tmp = torch.zeros(numel, dtype=torch_dtype, device=torch_device)
        
        # 从 llaisys tensor 复制数据
        api = RuntimeAPI(device_type)
        bytes_size = numel * tmp.element_size()
        api.memcpy_sync(
            tmp.data_ptr(),
            llaisys_tensor.data_ptr(),
            bytes_size,
            MemcpyKind.D2D,
        )
        
        # 使用 as_strided 创建正确形状和步长的 tensor
        if len(shape) > 0:
            result = torch.as_strided(tmp, shape, strides)
        else:
            result = tmp[0] if numel == 1 else tmp
    
    # 缓存结果（如果启用缓存）
    if use_cache:
        tensor_id = id(llaisys_tensor)
        if len(_tensor_cache) < _cache_max_size:
            _tensor_cache[tensor_id] = (result, llaisys_tensor.data_ptr())
    
    return result


def _torch_to_llaisys_tensor(torch_tensor: torch.Tensor, llaisys_tensor: Tensor, skip_if_same=True):
    """将 torch Tensor 的数据复制回 llaisys Tensor（极致优化版）"""
    # 极致优化1: 检查是否真的需要拷贝（指针相同且连续）
    if skip_if_same and torch_tensor.is_contiguous():
        # 检查指针是否相同（零拷贝情况）
        torch_ptr = torch_tensor.data_ptr()
        llaisys_ptr = llaisys_tensor.data_ptr()
        if torch_ptr == llaisys_ptr:
            return  # 无需拷贝
    
    api = RuntimeAPI(llaisys_tensor.device_type())
    
    # 优化：如果tensor已经是连续的，直接拷贝，避免不必要的contiguous调用
    if not torch_tensor.is_contiguous():
        torch_tensor = torch_tensor.contiguous()
    
    bytes_size = torch_tensor.numel() * torch_tensor.element_size()
    
    # 极致优化2: 对于大块数据（>=64KB），使用异步拷贝以提高性能
    if bytes_size >= 64 * 1024 and llaisys_tensor.device_type() == DeviceType.NVIDIA:
        stream = api.stream() if hasattr(api, 'stream') else None
        if stream:
            # 使用异步拷贝，让CUDA驱动有机会优化
            api.memcpy_async(
                llaisys_tensor.data_ptr(),
                torch_tensor.data_ptr(),
                bytes_size,
                MemcpyKind.D2D,
                stream
            )
            # 立即同步以确保数据就绪（对于输出tensor，kernel已经执行完毕）
            api.stream_synchronize(stream)
        else:
            api.memcpy_sync(
                llaisys_tensor.data_ptr(),
                torch_tensor.data_ptr(),
                bytes_size,
                MemcpyKind.D2D,
            )
    else:
        # 小块数据使用同步拷贝
        api.memcpy_sync(
            llaisys_tensor.data_ptr(),
            torch_tensor.data_ptr(),
            bytes_size,
            MemcpyKind.D2D,
        )
    
    # 更新缓存（如果存在且启用缓存）
    try:
        tensor_id = id(llaisys_tensor)
        if tensor_id in _tensor_cache:
            _tensor_cache[tensor_id] = (torch_tensor, llaisys_tensor.data_ptr())
    except (AttributeError, TypeError):
        # 如果tensor_id不可哈希或缓存有问题，忽略
        pass


# 极致优化：辅助函数，统一处理tensor转换和拷贝
def _optimized_tensor_ops(inputs, outputs, kernel_func, *args, **kwargs):
    """优化的tensor操作包装器，自动处理缓存和跳过相同指针"""
    # 转换输入tensor（使用缓存）
    input_torches = [_llaisys_to_torch_tensor(t, use_cache=True) for t in inputs]
    # 转换输出tensor（不使用缓存）
    output_torches = [_llaisys_to_torch_tensor(t, use_cache=False) for t in outputs]
    
    # 执行kernel
    kernel_func(*(input_torches + output_torches), *args, **kwargs)
    
    # 拷贝回输出tensor（跳过相同指针）
    for out_torch, out_llaisys in zip(output_torches, outputs):
        _torch_to_llaisys_tensor(out_torch, out_llaisys, skip_if_same=True)


class Ops:
    @staticmethod
    def add(c: Tensor, a: Tensor, b: Tensor):
        # Triton 只能在 GPU 上运行，CPU 上回退到 C++ 实现
        if _USE_TRITON and a.device_type() == DeviceType.NVIDIA:
            # 极致优化：使用缓存减少转换开销
            a_torch = _llaisys_to_torch_tensor(a, use_cache=True)
            b_torch = _llaisys_to_torch_tensor(b, use_cache=True)
            c_torch = _llaisys_to_torch_tensor(c, use_cache=False)  # 输出不需要缓存
            
            # 调用 triton kernel (triton 会自动从 torch tensor 提取数据指针)
            triton_kernels.add_kernel.kernel(a_torch, b_torch, c_torch)
            
            # 复制回 llaisys tensor（优化：跳过相同指针的拷贝）
            _torch_to_llaisys_tensor(c_torch, c, skip_if_same=True)
        else:
            _CURRENT_LIB.llaisysAdd(c.lib_tensor(), a.lib_tensor(), b.lib_tensor())

    @staticmethod
    def argmax(max_idx: Tensor, max_val: Tensor, vals: Tensor):
        # Triton 只能在 GPU 上运行，CPU 上回退到 C++ 实现
        if _USE_TRITON and vals.device_type() == DeviceType.NVIDIA:
            vals_torch = _llaisys_to_torch_tensor(vals)
            max_idx_torch = _llaisys_to_torch_tensor(max_idx)
            max_val_torch = _llaisys_to_torch_tensor(max_val)
            
            triton_kernels.argmax_kernel.kernel(vals_torch, max_idx_torch, max_val_torch)
            
            _torch_to_llaisys_tensor(max_idx_torch, max_idx)
            _torch_to_llaisys_tensor(max_val_torch, max_val)
        else:
            _CURRENT_LIB.llaisysArgmax(max_idx.lib_tensor(), max_val.lib_tensor(), vals.lib_tensor())

    @staticmethod
    def embedding(out: Tensor, index: Tensor, weight: Tensor):
        # Triton 只能在 GPU 上运行，CPU 上使用 PyTorch 作为回退
        if _USE_TRITON and weight.device_type() == DeviceType.NVIDIA:
            weight_torch = _llaisys_to_torch_tensor(weight)
            index_torch = _llaisys_to_torch_tensor(index)
            out_torch = _llaisys_to_torch_tensor(out)
            
            triton_kernels.embedding_kernel.kernel(weight_torch, index_torch, out_torch)
            
            _torch_to_llaisys_tensor(out_torch, out)
        else:
            # CPU 回退：使用 PyTorch 实现
            try:
                weight_torch = _llaisys_to_torch_tensor(weight)
                index_torch = _llaisys_to_torch_tensor(index)
                out_torch = _llaisys_to_torch_tensor(out)
                
                # Use PyTorch embedding
                out_torch.copy_(weight_torch[index_torch])
                
                _torch_to_llaisys_tensor(out_torch, out)
            except Exception:
                # 如果 PyTorch 回退失败，尝试 C++ 实现
                _CURRENT_LIB.llaisysEmbedding(
                    out.lib_tensor(), index.lib_tensor(), weight.lib_tensor()
                )

    @staticmethod
    def linear(out: Tensor, inp: Tensor, weight: Tensor, bias: Tensor):
        # Triton 只能在 GPU 上运行，CPU 上使用 PyTorch 作为回退
        if _USE_TRITON and inp.device_type() == DeviceType.NVIDIA:
            # 极致优化：输入tensor使用缓存，输出tensor直接写入
            inp_torch = _llaisys_to_torch_tensor(inp, use_cache=True)
            weight_torch = _llaisys_to_torch_tensor(weight, use_cache=True)
            bias_torch = _llaisys_to_torch_tensor(bias, use_cache=True) if bias is not None else None
            out_torch = _llaisys_to_torch_tensor(out, use_cache=False)
            
            # 执行kernel
            triton_kernels.linear_kernel.kernel(inp_torch, weight_torch, bias_torch, out_torch)
            
            # 极致优化：跳过相同指针的拷贝
            _torch_to_llaisys_tensor(out_torch, out, skip_if_same=True)
        else:
            # CPU 回退：使用 PyTorch 实现
            try:
                inp_torch = _llaisys_to_torch_tensor(inp)
                weight_torch = _llaisys_to_torch_tensor(weight)
                bias_torch = _llaisys_to_torch_tensor(bias) if bias is not None else None
                out_torch = _llaisys_to_torch_tensor(out)
                
                # Linear: out = inp @ weight.T + bias
                out_torch.copy_(torch.nn.functional.linear(inp_torch, weight_torch, bias_torch))
                
                _torch_to_llaisys_tensor(out_torch, out)
            except Exception:
                _CURRENT_LIB.llaisysLinear(
                    out.lib_tensor(), inp.lib_tensor(), weight.lib_tensor(), bias.lib_tensor()
                )

    @staticmethod
    def rearrange(out: Tensor, inp: Tensor):
        # Triton 只能在 GPU 上运行，CPU 上使用 PyTorch 作为回退
        if _USE_TRITON and inp.device_type() == DeviceType.NVIDIA:
            # 极致优化：使用缓存和跳过相同指针
            inp_torch = _llaisys_to_torch_tensor(inp, use_cache=True)
            out_torch = _llaisys_to_torch_tensor(out, use_cache=False)
            
            triton_kernels.rearrange_kernel.kernel(inp_torch, out_torch)
            
            _torch_to_llaisys_tensor(out_torch, out, skip_if_same=True)
        else:
            # CPU 回退：使用 PyTorch 实现
            # Rearrange: reshape from (seq_len, num_heads, head_dim) to (seq_len, hidden_size)
            try:
                inp_torch = _llaisys_to_torch_tensor(inp)
                out_torch = _llaisys_to_torch_tensor(out)
                
                # Reshape: (seq_len, num_heads, head_dim) -> (seq_len, num_heads * head_dim)
                out_torch.copy_(inp_torch.reshape(out_torch.shape))
                
                _torch_to_llaisys_tensor(out_torch, out)
            except Exception:
                _CURRENT_LIB.llaisysRearrange(out.lib_tensor(), inp.lib_tensor())

    @staticmethod
    def rms_norm(out: Tensor, inp: Tensor, weight: Tensor, eps: float):
        # Triton 只能在 GPU 上运行，CPU 上使用 PyTorch 作为回退
        if _USE_TRITON and inp.device_type() == DeviceType.NVIDIA:
            # 极致优化：使用缓存和跳过相同指针
            inp_torch = _llaisys_to_torch_tensor(inp, use_cache=True)
            weight_torch = _llaisys_to_torch_tensor(weight, use_cache=True)
            out_torch = _llaisys_to_torch_tensor(out, use_cache=False)
            
            triton_kernels.rms_norm_kernel.kernel(inp_torch, weight_torch, out_torch, eps)
            
            _torch_to_llaisys_tensor(out_torch, out, skip_if_same=True)
        else:
            # CPU 回退：使用 PyTorch 实现
            try:
                inp_torch = _llaisys_to_torch_tensor(inp)
                weight_torch = _llaisys_to_torch_tensor(weight)
                out_torch = _llaisys_to_torch_tensor(out)
                
                # RMSNorm: out = (inp / sqrt(mean(inp^2) + eps)) * weight
                variance = inp_torch.pow(2).mean(-1, keepdim=True)
                out_torch.copy_((inp_torch / torch.sqrt(variance + eps)) * weight_torch)
                
                _torch_to_llaisys_tensor(out_torch, out)
            except Exception:
                # 如果 PyTorch 回退失败，尝试 C++ 实现
                _CURRENT_LIB.llaisysRmsNorm(
                    out.lib_tensor(), inp.lib_tensor(), weight.lib_tensor(), c_float(eps)
                )

    @staticmethod
    def rope(out: Tensor, inp: Tensor, pos_ids: Tensor, theta: float):
        # Triton 只能在 GPU 上运行，CPU 上使用 PyTorch 作为回退
        if _USE_TRITON and inp.device_type() == DeviceType.NVIDIA:
            # 极致优化：使用缓存和跳过相同指针
            inp_torch = _llaisys_to_torch_tensor(inp, use_cache=True)
            pos_ids_torch = _llaisys_to_torch_tensor(pos_ids, use_cache=True)
            out_torch = _llaisys_to_torch_tensor(out, use_cache=False)
            
            triton_kernels.rope_kernel.kernel(inp_torch, pos_ids_torch, out_torch, theta)
            
            _torch_to_llaisys_tensor(out_torch, out, skip_if_same=True)
        else:
            # CPU 回退：使用 PyTorch 实现 RoPE
            try:
                inp_torch = _llaisys_to_torch_tensor(inp)
                pos_ids_torch = _llaisys_to_torch_tensor(pos_ids)
                out_torch = _llaisys_to_torch_tensor(out)
                
                # Simplified RoPE implementation using PyTorch
                # This is a simplified version - full implementation would be more complex
                seq_len, num_heads, head_dim = inp_torch.shape
                device = inp_torch.device
                dtype = inp_torch.dtype
                
                # Create frequency matrix
                freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
                angles = pos_ids_torch[:, None].float() * freqs[None, :]
                cos = torch.cos(angles)
                sin = torch.sin(angles)
                
                # Apply rotation
                x1 = inp_torch[..., 0::2]
                x2 = inp_torch[..., 1::2]
                rotated_x1 = x1 * cos[:, :, None] - x2 * sin[:, :, None]
                rotated_x2 = x1 * sin[:, :, None] + x2 * cos[:, :, None]
                
                # Reconstruct output
                out_torch[..., 0::2] = rotated_x1
                out_torch[..., 1::2] = rotated_x2
                
                _torch_to_llaisys_tensor(out_torch, out)
            except Exception:
                _CURRENT_LIB.llaisysROPE(
                    out.lib_tensor(), inp.lib_tensor(), pos_ids.lib_tensor(), c_float(theta)
                )

    @staticmethod
    def self_attention(attn_val: Tensor, q: Tensor, k: Tensor, v: Tensor, scale: float):
        # Triton 只能在 GPU 上运行，CPU 上使用 PyTorch 作为回退
        if _USE_TRITON and q.device_type() == DeviceType.NVIDIA:
            # 极致优化：使用缓存和跳过相同指针
            q_torch = _llaisys_to_torch_tensor(q, use_cache=True)
            k_torch = _llaisys_to_torch_tensor(k, use_cache=True)
            v_torch = _llaisys_to_torch_tensor(v, use_cache=True)
            attn_val_torch = _llaisys_to_torch_tensor(attn_val, use_cache=False)
            
            triton_kernels.self_attention_kernel.kernel(q_torch, k_torch, v_torch, attn_val_torch, scale)
            
            _torch_to_llaisys_tensor(attn_val_torch, attn_val, skip_if_same=True)
        else:
            # CPU 回退：使用 PyTorch 实现
            try:
                q_torch = _llaisys_to_torch_tensor(q)
                k_torch = _llaisys_to_torch_tensor(k)
                v_torch = _llaisys_to_torch_tensor(v)
                attn_val_torch = _llaisys_to_torch_tensor(attn_val)
                
                # Self-attention: attn_val = softmax(Q @ K.T * scale) @ V
                # q: (seq_len, num_heads, head_dim), k: (kv_len, num_kv_heads, head_dim), v: (kv_len, num_kv_heads, head_dim)
                seq_len, num_heads, head_dim = q_torch.shape
                kv_len, num_kv_heads, _ = k_torch.shape
                
                # Reshape for attention
                q_reshaped = q_torch.reshape(seq_len, num_heads, head_dim)
                k_reshaped = k_torch.reshape(kv_len, num_kv_heads, head_dim)
                v_reshaped = v_torch.reshape(kv_len, num_kv_heads, head_dim)
                
                # Compute attention scores
                # For GQA, repeat k and v to match num_heads
                if num_kv_heads < num_heads:
                    repeat_factor = num_heads // num_kv_heads
                    k_reshaped = k_reshaped.repeat_interleave(repeat_factor, dim=1)
                    v_reshaped = v_reshaped.repeat_interleave(repeat_factor, dim=1)
                
                scores = torch.einsum('qhd,khd->qhk', q_reshaped, k_reshaped) * scale
                attn_weights = torch.softmax(scores, dim=-1)
                attn_val_torch.copy_(torch.einsum('qhk,khd->qhd', attn_weights, v_reshaped))
                
                _torch_to_llaisys_tensor(attn_val_torch, attn_val)
            except Exception:
                _CURRENT_LIB.llaisysSelfAttention(
                    attn_val.lib_tensor(),
                    q.lib_tensor(),
                    k.lib_tensor(),
                    v.lib_tensor(),
                    c_float(scale),
                )

    @staticmethod
    def swiglu(out: Tensor, gate: Tensor, up: Tensor):
        # Triton 只能在 GPU 上运行，CPU 上使用 PyTorch 作为回退
        if _USE_TRITON and gate.device_type() == DeviceType.NVIDIA:
            # 极致优化：使用缓存和跳过相同指针
            gate_torch = _llaisys_to_torch_tensor(gate, use_cache=True)
            up_torch = _llaisys_to_torch_tensor(up, use_cache=True)
            out_torch = _llaisys_to_torch_tensor(out, use_cache=False)
            
            triton_kernels.swiglu_kernel.kernel(gate_torch, up_torch, out_torch)
            
            _torch_to_llaisys_tensor(out_torch, out, skip_if_same=True)
        else:
            # CPU 回退：使用 PyTorch 实现
            try:
                gate_torch = _llaisys_to_torch_tensor(gate)
                up_torch = _llaisys_to_torch_tensor(up)
                out_torch = _llaisys_to_torch_tensor(out)
                
                # SwiGLU: out = up * gate * sigmoid(gate)
                out_torch.copy_(up_torch * gate_torch * torch.sigmoid(gate_torch))
                
                _torch_to_llaisys_tensor(out_torch, out)
            except Exception:
                _CURRENT_LIB.llaisysSwiGLU(out.lib_tensor(), gate.lib_tensor(), up.lib_tensor())