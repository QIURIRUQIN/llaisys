import triton
import triton.language as tl
import math

@triton.jit
def rope_kernel(
    input_ptr, pos_ids_ptr, output_ptr,
    seq_len, n_head, d, theta,
    stride_im, stride_in, stride_ih,
    stride_om, stride_on, stride_oh,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    head_id = pid // seq_len
    pos = pid % seq_len
    
    if head_id >= n_head or pos >= seq_len:
        return
    
    pos_id = tl.load(pos_ids_ptr + pos)
    d_half = d // 2
    
    for i in range(0, d_half, BLOCK_SIZE):
        idx = i + tl.arange(0, BLOCK_SIZE)
        mask = idx < d_half
        
        # Load input
        a_idx = idx
        b_idx = idx + d_half
        
        a_offset = pos * stride_im + head_id * stride_in + a_idx * stride_ih
        b_offset = pos * stride_im + head_id * stride_in + b_idx * stride_ih
        
        a = tl.load(input_ptr + a_offset, mask=mask, other=0.0)
        b = tl.load(input_ptr + b_offset, mask=mask, other=0.0)
        
        # Convert to float32 for computation
        a = a.to(tl.float32)
        b = b.to(tl.float32)
        
        # Use float64 for high-precision frequency computation to reduce numerical errors
        # Match PyTorch: freqs = positions / (theta ** (2 * i / head_dim))
        pos_id_f64 = pos_id.to(tl.float64)
        idx_f64 = idx.to(tl.float64)
        exponent = 2.0 * idx_f64 / d
        
        # Use log2/exp2 for better numerical precision when computing powers
        # theta^exponent = 2^(exponent * log2(theta))
        # Ensure theta is treated as float
        theta_f64 = theta + 0.0  # Ensure float type
        log2_theta = tl.log2(theta_f64).to(tl.float64)
        log2_power = exponent * log2_theta
        theta_power = tl.exp2(log2_power)  # Compute theta^exponent in float64
        
        # Compute frequency in float64, then convert to float32
        freqs_f64 = pos_id_f64 / theta_power
        angle = freqs_f64.to(tl.float32)
        
        cos_val = tl.cos(angle)
        sin_val = tl.sin(angle)
        
        # Apply RoPE
        a_out = a * cos_val - b * sin_val
        b_out = b * cos_val + a * sin_val
        
        # Store output
        a_out_offset = pos * stride_om + head_id * stride_on + a_idx * stride_oh
        b_out_offset = pos * stride_om + head_id * stride_on + b_idx * stride_oh
        
        tl.store(output_ptr + a_out_offset, a_out, mask=mask)
        tl.store(output_ptr + b_out_offset, b_out, mask=mask)

def kernel(input, pos_ids, output, theta, BLOCK_SIZE=64):
    seq_len, n_head, d = input.shape
    grid = (seq_len * n_head,)
    rope_kernel[grid](
        input, pos_ids, output,
        seq_len, n_head, d, theta,
        input.stride(0), input.stride(1), input.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_SIZE=BLOCK_SIZE,
    )
