import triton
import triton.language as tl

@triton.jit
def swiglu_kernel(
    gate_ptr, up_ptr, output_ptr,
    seq_len, intermediate_size,
    stride_gm, stride_gn,
    stride_um, stride_un,
    stride_om, stride_on,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row = pid
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < intermediate_size
    
    if row >= seq_len:
        return
    
    gate_offset = row * stride_gm + cols * stride_gn
    up_offset = row * stride_um + cols * stride_un
    output_offset = row * stride_om + cols * stride_on
    
    gate = tl.load(gate_ptr + gate_offset, mask=mask, other=0.0)
    up = tl.load(up_ptr + up_offset, mask=mask, other=0.0)
    
    # SwiGLU: output = up * gate * sigmoid(gate)
    # This is equivalent to: up * (gate / (1 + exp(-gate)))
    # Compute in float32 for precision (like PyTorch's gate.float())
    gate_f32 = gate.to(tl.float32)
    up_f32 = up.to(tl.float32)
    
    # sigmoid(gate) = 1 / (1 + exp(-gate))
    sigmoid_gate = 1.0 / (1.0 + tl.exp(-gate_f32))
    
    # Compute result in float32: up * gate * sigmoid(gate)
    result_f32 = up_f32 * gate_f32 * sigmoid_gate
    
    # Convert back to original dtype for storage
    output = result_f32.to(gate.dtype)
    
    tl.store(output_ptr + output_offset, output, mask=mask)

def kernel(gate, up, output, BLOCK_SIZE=1024):
    seq_len, intermediate_size = gate.shape
    # Ensure BLOCK_SIZE is a power of 2 for tl.arange, and at least as large as intermediate_size
    BLOCK_SIZE = max(triton.next_power_of_2(intermediate_size), 32)
    grid = (seq_len,)
    swiglu_kernel[grid](
        gate, up, output,
        seq_len, intermediate_size,
        gate.stride(0), gate.stride(1),
        up.stride(0), up.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
