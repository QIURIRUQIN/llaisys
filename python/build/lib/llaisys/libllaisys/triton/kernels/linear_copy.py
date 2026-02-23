import triton
import triton.language as tl

@triton.jit
def linear_kernel_large(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K,
    stride_im, stride_in,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Kernel for large matrices using tl.dot (requires M,N,K >= 16)"""
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    accumulator = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        k_offs = k + offs_k
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        b_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        
        # Load input: x[offs_m, k_offs] shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a = tl.load(input_ptr + offs_m[:, None] * stride_im + k_offs[None, :] * stride_in, mask=a_mask, other=0.0)
        b = tl.load(weight_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk, mask=b_mask, other=0.0)

        a_f32 = a.to(tl.float32)
        b_f32 = b.to(tl.float32)

        accumulator += tl.dot(a_f32, b_f32, allow_tf32=False)
    
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        accumulator += bias[None, :]
    
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, accumulator, mask=c_mask)

@triton.jit
def linear_kernel_small(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K,
    stride_im, stride_in,
    stride_wn, stride_wk,  # weight is (N, K), we need w.T which is (K, N)
    stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Kernel for small matrices using manual loop (for when tl.dot cannot be used)"""
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    accumulator = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        k_offs = k + offs_k
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        b_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        
        a = tl.load(input_ptr + offs_m[:, None] * stride_im + k_offs[None, :] * stride_in, mask=a_mask, other=0.0)
        b = tl.load(weight_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk, mask=b_mask, other=0.0)

        a_f32 = a.to(tl.float32)  # Ensure float32 for better precision
        b_f32 = b.to(tl.float32)
        a_expanded = a_f32[:, :, None]
        b_expanded = b_f32[None, :, :]
        product = a_expanded * b_expanded
        k_sum = tl.sum(product, axis=1)
        accumulator += k_sum
    
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        accumulator += bias[None, :]
    
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, accumulator, mask=c_mask)

def kernel(input, weight, bias, output, BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32):
    """
    Linear layer kernel: output = input @ weight.T + bias
    
    Args:
        input: (M, K) tensor
        weight: (N, K) tensor - 注意：原始weight格式，内部会转置处理
        bias: (N,) tensor or None
        output: (M, N) tensor
    """
    M, K = input.shape
    N = weight.shape[0]
    # weight: [N, K]

    if M <= 64 and N <= 64:
        BLOCK_SIZE_M = min(BLOCK_SIZE_M, triton.next_power_of_2(M))
        BLOCK_SIZE_N = min(BLOCK_SIZE_N, triton.next_power_of_2(N))
        BLOCK_SIZE_K = min(BLOCK_SIZE_K, triton.next_power_of_2(K))
    
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    
    use_large_kernel = (BLOCK_SIZE_M >= 16 and BLOCK_SIZE_N >= 16 and BLOCK_SIZE_K >= 16)
    
    if use_large_kernel:
        linear_kernel_large[grid](
            input, weight, bias, output,
            M, N, K,
            input.stride(0), input.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
    else:
        linear_kernel_small[grid](
            input, weight, bias, output,
            M, N, K,
            input.stride(0), input.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
