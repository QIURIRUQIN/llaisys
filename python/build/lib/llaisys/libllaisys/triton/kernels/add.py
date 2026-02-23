import triton
import triton.language as tl

@triton.jit
def add_kernel(
    input_ptr, other_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """优化的add kernel，使用更大的block size"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    input = tl.load(input_ptr + offsets, mask=mask)
    other = tl.load(other_ptr + offsets, mask=mask)
    output = input + other
    tl.store(output_ptr + offsets, output, mask=mask)

def kernel(input, other, output, BLOCK_SIZE=8192):
    """
    极致优化的add kernel，使用更大的block size以获得更好的性能
    
    Args:
        input: 输入tensor
        other: 另一个输入tensor
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
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    add_kernel[grid](
        input, other, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
