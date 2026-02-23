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
    GROUP_SIZE_M: tl.constexpr,  # L2 cache swizzling 参数
    ALLOW_TF32: tl.constexpr,
):
    """Optimized kernel for large matrices using tl.dot"""
    pid = tl.program_id(axis=0)
    
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # 优化1: L2 Cache Swizzling - 当block数量足够时启用
    if GROUP_SIZE_M > 1 and num_pid_m > GROUP_SIZE_M:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # 优化2: 预计算基础指针，循环中只做增量更新
    input_block_ptr = input_ptr + offs_m[:, None] * stride_im + offs_k[None, :] * stride_in
    weight_block_ptr = weight_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
    
    accumulator = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    
    # 优化3: 主循环 - 使用指针递增代替重新计算
    for k in range(0, K, BLOCK_SIZE_K):
        k_remaining = K - k
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_n[None, :] < N) & (offs_k[:, None] < k_remaining)
        
        # 加载数据块
        a = tl.load(input_block_ptr, mask=a_mask, other=0.0)
        b = tl.load(weight_block_ptr, mask=b_mask, other=0.0)
        
        # 优化4: 可选TF32加速 (Ampere及以上GPU)，并使用累加器形式
        accumulator = tl.dot(a, b, acc=accumulator, allow_tf32=ALLOW_TF32)
        
        # 指针递增
        input_block_ptr += BLOCK_SIZE_K * stride_in
        weight_block_ptr += BLOCK_SIZE_K * stride_wk
    
    # 添加bias
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        accumulator += bias[None, :]
    
    # 存储结果
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    output_block_ptr = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(output_block_ptr, accumulator, mask=c_mask)


@triton.jit
def linear_kernel_small(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K,
    stride_im, stride_in,
    stride_wn, stride_wk,
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

        a_f32 = a.to(tl.float32)
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


def kernel(input, weight, bias, output, BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32):
    """
    Linear layer kernel: output = input @ weight.T + bias
    
    Args:
        input: (M, K) tensor
        weight: (N, K) tensor
        bias: (N,) tensor or None
        output: (M, N) tensor
    """
    M, K = input.shape
    N = weight.shape[0]
    
    # 依据规模选择参数
    # 目标：大矩阵更大tile，K大时更大BLOCK_SIZE_K；小矩阵保守设置
    if (M >= 256 and N >= 256) or K >= 256:
        cfg = dict(BM=128, BN=128, BK=64, GROUP=4, WARPS=4, STAGES=3, ALLOW_TF32=True)
    elif M <= 128 and N <= 128:
        cfg = dict(BM=64, BN=128, BK=32, GROUP=2, WARPS=4, STAGES=2, ALLOW_TF32=True)
    else:
        cfg = dict(BM=128, BN=64, BK=32, GROUP=4, WARPS=4, STAGES=3, ALLOW_TF32=True)

    BLOCK_SIZE_M = cfg["BM"]
    BLOCK_SIZE_N = cfg["BN"]
    BLOCK_SIZE_K = cfg["BK"]
    group_size_m = cfg["GROUP"]
    num_warps = cfg["WARPS"]
    num_stages = cfg["STAGES"]
    allow_tf32 = cfg["ALLOW_TF32"]

    # 小矩阵再做一次收缩
    if M <= 64 and N <= 64:
        BLOCK_SIZE_M = min(BLOCK_SIZE_M, triton.next_power_of_2(M))
        BLOCK_SIZE_N = min(BLOCK_SIZE_N, triton.next_power_of_2(N))
        BLOCK_SIZE_K = min(BLOCK_SIZE_K, triton.next_power_of_2(K))
        group_size_m = 1  # block 数少时禁用 swizzling
        num_warps = 2
        num_stages = 2
        allow_tf32 = False  # 很小矩阵时 TF32 不一定有优势
    
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    
    use_large_kernel = (BLOCK_SIZE_M >= 16 and BLOCK_SIZE_N >= 16 and BLOCK_SIZE_K >= 16)
    
    if use_large_kernel:
        # 优化5: 配置num_warps和num_stages
        # - num_warps: 更多warps可以更好地隐藏内存延迟
        # - num_stages: 软件流水线阶段数，用于预取数据
        linear_kernel_large[grid](
            input, weight, bias, output,
            M, N, K,
            input.stride(0), input.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=group_size_m,
            ALLOW_TF32=allow_tf32,
            num_warps=num_warps,
            num_stages=num_stages,
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
