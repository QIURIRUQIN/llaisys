import triton
import triton.language as tl

@triton.jit
def embedding_kernel(
    weight_ptr, index_ptr, output_ptr,
    vocab_size, hidden_size, seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < seq_len
    
    indices = tl.load(index_ptr + offsets, mask=mask, other=0)
    
    for i in range(BLOCK_SIZE):
        if i + block_start < seq_len:
            idx = tl.load(index_ptr + block_start + i)
            if idx >= 0 and idx < vocab_size:
                weight_offset = idx * hidden_size
                output_offset = (block_start + i) * hidden_size
                for j in range(hidden_size):
                    val = tl.load(weight_ptr + weight_offset + j)
                    tl.store(output_ptr + output_offset + j, val)

def kernel(weight, index, output, BLOCK_SIZE=1024):
    seq_len = index.numel()
    vocab_size = weight.shape[0]
    hidden_size = weight.shape[1]
    grid = (triton.cdiv(seq_len, BLOCK_SIZE),)
    embedding_kernel[grid](
        weight, index, output,
        vocab_size, hidden_size, seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )

