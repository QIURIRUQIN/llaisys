import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    input_ptr, weight_ptr, output_ptr,
    seq_len, hidden_size, eps,
    stride_im, stride_in,
    stride_om, stride_on,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    
    # First pass: compute sum of x^2 across all blocks for this row
    sum_squared = 0.0
    for block_start in range(0, hidden_size, BLOCK_SIZE):
        cols = tl.arange(0, BLOCK_SIZE) + block_start
        mask = cols < hidden_size
        input_offsets = row * stride_im + cols * stride_in
        x = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
        x_squared = x * x
        sum_squared += tl.sum(x_squared, axis=0)
    
    # Compute mean and rsqrt
    mean_squared = sum_squared / hidden_size
    mean_squared_eps = mean_squared + eps
    rsqrt_val = tl.rsqrt(mean_squared_eps)  # 1/sqrt(mean(x^2) + eps)
    
    # Second pass: normalize and scale all blocks for this row
    for block_start in range(0, hidden_size, BLOCK_SIZE):
        cols = tl.arange(0, BLOCK_SIZE) + block_start
        mask = cols < hidden_size
        input_offsets = row * stride_im + cols * stride_in
        output_offsets = row * stride_om + cols * stride_on
        
        x = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        
        # Normalize and scale
        x_norm = x * rsqrt_val
        output = x_norm * weight
        
        tl.store(output_ptr + output_offsets, output, mask=mask)

def kernel(input, weight, output, eps, BLOCK_SIZE=1024):
    seq_len, hidden_size = input.shape
    grid = (seq_len,)
    rms_norm_kernel[grid](
        input, weight, output,
        seq_len, hidden_size, eps,
        input.stride(0), input.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
