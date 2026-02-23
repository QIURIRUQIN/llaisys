import triton
import triton.language as tl

@triton.jit
def rearrange_kernel(
    input_ptr, output_ptr,
    n_elements,
    input_stride, output_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """优化的rearrange kernel，支持真实的stride"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 使用真实的stride进行加载和存储
    input_vals = tl.load(input_ptr + offsets * input_stride, mask=mask)
    tl.store(output_ptr + offsets * output_stride, input_vals, mask=mask)

def kernel(input, output, BLOCK_SIZE=8192):
    """
    极致优化的rearrange kernel，使用真实的stride和自适应block size
    
    Args:
        input: 输入tensor
        output: 输出tensor
        BLOCK_SIZE: block大小，默认8192（极致优化）
    """
    n_elements = input.numel()
    
    # 自适应block size：对于小tensor，使用较小的block
    if n_elements < 8192:
        BLOCK_SIZE = min(BLOCK_SIZE, triton.next_power_of_2(n_elements))
        num_warps = 2
    elif n_elements < 65536:
        BLOCK_SIZE = 4096
        num_warps = 4
    else:
        BLOCK_SIZE = 8192
        num_warps = 8
    
    # 如果tensor是连续的，stride为1，否则使用真实的stride
    if input.is_contiguous() and output.is_contiguous():
        input_stride = 1
        output_stride = 1
    else:
        # 对于非连续tensor，使用第一个维度的stride
        # 注意：这是一个简化，完整的实现需要处理多维stride
        input_stride = input.stride(0) if input.dim() > 0 else 1
        output_stride = output.stride(0) if output.dim() > 0 else 1
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    rearrange_kernel[grid](
        input, output,
        n_elements,
        input_stride, output_stride,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

