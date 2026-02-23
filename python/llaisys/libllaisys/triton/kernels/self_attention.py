"""Triton kernel for self-attention operation.

Flash Attention style implementation with:
- Online softmax for numerical stability
- Block-wise processing for memory efficiency
- Causal masking support
- Separate kernels for prefill and decode stages
"""
import torch
import triton
import triton.language as tl

@triton.jit
def self_attention_prefill_kernel(
    q_ptr, k_ptr, v_ptr, attn_val_ptr,
    q_len, kv_len, n_head, nkv_head, d, dv, scale,
    stride_qm, stride_qn, stride_qd,
    stride_km, stride_kn, stride_kd,
    stride_vm, stride_vn, stride_vd,
    stride_om, stride_on, stride_od,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    USE_DOT: tl.constexpr,
    OUT_DTYPE: tl.constexpr
):
    """Flash Attention style kernel for prefill stage (multiple query positions).
    
    Each program processes a block of query positions for one head.
    Uses online softmax to avoid storing full attention matrix.
    """
    pid_m = tl.program_id(0)  # Query block index
    pid_h = tl.program_id(1)  # Head index (query)

    log2_e = 1.44269504
    
    if pid_h >= n_head:
        return
        
    assert n_head % nkv_head == 0, "n_head must be divisible by nkv_head"
    group_size = n_head // nkv_head
    kv_head_id = pid_h // group_size
    
    # Base offsets
    q_head_off = pid_h * stride_qn
    k_head_off = kv_head_id * stride_kn
    v_head_off = kv_head_id * stride_vn
    out_head_off = pid_h * stride_on
    
    # Load Q block: (BLOCK_SIZE_M, HEAD_DIM)
    q_dim = tl.arange(0, HEAD_DIM)
    q_mask = q_dim < d
    # q_offsets = q_head_off + offs_m[:, None] * stride_qm + q_dim[None, :] * stride_qd
    # q_mask_full = mask_m[:, None] & q_mask[None, :]
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr+q_head_off,
        shape=(q_len, d),
        strides=(stride_qm, stride_qd),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, HEAD_DIM),
        order=(1, 0)
    )
    q = tl.load(q_block_ptr, boundary_check=(0, 1)).to(tl.float32)

    # 进行逻辑转置
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr+k_head_off,
        shape=(d, kv_len),
        strides=(stride_kd, stride_km),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_N),
        order=(0, 1)
    )

    v_block_ptr = tl.make_block_ptr(
        base=v_ptr+v_head_off,
        shape=(kv_len, dv),
        strides=(stride_vm, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_N, HEAD_DIM),
        order=(1, 0)
    )
    
    acc = tl.zeros((BLOCK_SIZE_M, HEAD_DIM), dtype=tl.float32)
    m_i = tl.full((BLOCK_SIZE_M,), float("-inf"), dtype=tl.float32)
    d_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Iterate over KV blocks
    for kv_block_start in range(0, kv_len, BLOCK_SIZE_N):
        k = tl.load(k_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        v = tl.load(v_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        
        # Compute QK^T: (BLOCK_SIZE_M, BLOCK_SIZE_N)
        # Use tl.dot for better precision when d >= 16, otherwise manual multiplication
        if USE_DOT:
            # Use tl.dot with IEEE precision for better numerical stability
            attention_scores = tl.dot(q, k, input_precision="ieee") * scale # (BLOCK_SIZE_M, BLOCK_SIZE_N)
        else:
            q_expanded = q[:, :, None]
            k_expanded = k[None, :, :]
            product = q_expanded * k_expanded
            product = tl.where(q_mask[None, :, None], product, 0.0)
            attention_scores = tl.sum(product, axis=1) * scale
        
        # boundary mask
        kv_idx = kv_block_start + tl.arange(0, BLOCK_SIZE_N)
        kv_mask = kv_idx < kv_len
        # casual mask
        causal_limits = BLOCK_SIZE_M * pid_m + tl.arange(0, BLOCK_SIZE_M) + (kv_len - q_len) + 1
        casual_mask = kv_idx[None, :] < causal_limits[:, None]

        mask_m = (BLOCK_SIZE_M * pid_m + tl.arange(0, BLOCK_SIZE_M)) < q_len

        full_mask = mask_m[:, None] & kv_mask[None, :] & casual_mask
        # Apply mask: set masked positions to -inf so that exp -> 0
        masked_attention_scores = tl.where(full_mask, attention_scores, float("-inf"))
        
        # Online softmax update in float32
        m_i_new = tl.maximum(m_i, tl.max(masked_attention_scores, axis=1))
        alpha = tl.exp2(log2_e * (m_i - m_i_new))
        exp_attention_scores = tl.exp2(log2_e * (masked_attention_scores - m_i_new[:, None]))
        d_i_k = tl.sum(exp_attention_scores, axis=1)
        
        if USE_DOT and dv >= 16 and BLOCK_SIZE_M >= 16:
            # Use tl.dot with IEEE precision
            o = tl.dot(exp_attention_scores, v, input_precision="ieee")  # (BLOCK_SIZE_M, HEAD_DIM)
        else:
            exp_scores_expanded = exp_attention_scores[:, :, None]  # (M, N, 1)
            v_expanded = v[None, :, :]  # (1, N, Dv)
            o = tl.sum(exp_scores_expanded * v_expanded, axis=1)
        
        acc = acc * alpha[:, None] + o
        # acc = tl.fma(acc, alpha[:, None], o)
        
        # Update running statistics
        m_i = m_i_new
        # d_i = tl.fma(d_i, alpha, d_i_k)
        d_i = d_i * alpha + d_i_k

        k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_SIZE_N))
        v_block_ptr = tl.advance(v_block_ptr, (BLOCK_SIZE_N, 0))
    
    # Final normalization
    acc = acc / d_i[:, None]
    
    out_block_ptr = tl.make_block_ptr(
        base=attn_val_ptr+out_head_off,
        shape=(q_len, dv),
        strides=(stride_om, stride_od),
        offsets=(BLOCK_SIZE_M * pid_m, 0),
        block_shape=(BLOCK_SIZE_M, HEAD_DIM),
        order=(1, 0),
    )
    
    tl.store(out_block_ptr, acc.to(OUT_DTYPE), boundary_check=(0, 1))

@triton.jit
def self_attention_decode_kernel(
    q_ptr, k_ptr, v_ptr, attn_val_ptr,
    seq_len, total_len, n_head, nkv_head, d, dv, scale,
    stride_qm, stride_qn, stride_qd,
    stride_km, stride_kn, stride_kd,
    stride_vm, stride_vn, stride_vd,
    stride_om, stride_on, stride_od,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    USE_DOT: tl.constexpr,  # Whether to use tl.dot (d >= 16)
    OUT_DTYPE: tl.constexpr
):
    """Flash Attention style kernel for decode stage (single query position).
    
    For decode stage, seq_len should be 1.
    Processes all kv positions in blocks using online softmax.
    """
    pid_m = tl.program_id(0)  # Query block index (should be 0 for decode)
    pid_h = tl.program_id(1)  # Head index
    
    if pid_h >= n_head or pid_m * BLOCK_SIZE_M >= seq_len:
        return
    
    # Map query head to KV head for GQA
    assert n_head % nkv_head == 0, "n_head must be divisible by nkv_head"
    group_size = n_head // nkv_head
    kv_head_id = pid_h // group_size
    
    # Query positions (should be just 1 for decode)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_m < seq_len
    
    # Base offsets
    q_head_off = pid_h * stride_qn
    k_head_off = kv_head_id * stride_kn
    v_head_off = kv_head_id * stride_vn
    out_head_off = pid_h * stride_on
    
    # Load Q: (BLOCK_SIZE_M, HEAD_DIM)
    q_dim = tl.arange(0, HEAD_DIM)
    q_mask = q_dim < d
    q_offsets = q_head_off + offs_m[:, None] * stride_qm + q_dim[None, :] * stride_qd
    q_mask_full = mask_m[:, None] & q_mask[None, :]
    q = tl.load(q_ptr + q_offsets, mask=q_mask_full, other=0.0).to(tl.float32)
    
    # Initialize online softmax state and output accumulator
    acc = tl.zeros((BLOCK_SIZE_M, HEAD_DIM), dtype=tl.float32)
    m_i = tl.full((BLOCK_SIZE_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Process all kv positions in blocks
    for kv_block_start in range(0, total_len, BLOCK_SIZE_N):
        kv_idx = kv_block_start + tl.arange(0, BLOCK_SIZE_N)
        kv_mask = kv_idx < total_len
        
        # Load K block: (HEAD_DIM, BLOCK_SIZE_N)
        k_dim = tl.arange(0, HEAD_DIM)
        k_mask = k_dim < d
        k_offsets = k_head_off + k_dim[:, None] * stride_kd + kv_idx[None, :] * stride_km
        k_mask_full = k_mask[:, None] & kv_mask[None, :]
        k = tl.load(k_ptr + k_offsets, mask=k_mask_full, other=0.0).to(tl.float32)
        
        # Compute QK^T: (BLOCK_SIZE_M, BLOCK_SIZE_N)
        # Use tl.dot for better precision when d >= 16, otherwise manual multiplication
        if USE_DOT:
            # Use tl.dot with IEEE precision for better numerical stability
            qk = tl.dot(q, k, input_precision="ieee") * scale  # (BLOCK_SIZE_M, BLOCK_SIZE_N)
        else:
            # Manual matrix multiplication for small dimensions
            q_expanded = q[:, :, None]  # (BLOCK_SIZE_M, HEAD_DIM, 1)
            k_expanded = k[None, :, :]  # (1, HEAD_DIM, BLOCK_SIZE_N)
            product = q_expanded * k_expanded  # (BLOCK_SIZE_M, HEAD_DIM, BLOCK_SIZE_N)
            # Apply mask for valid d dimensions
            d_mask_expanded = q_mask[None, :, None]  # (1, HEAD_DIM, 1)
            product = tl.where(d_mask_expanded, product, 0.0)
            qk = tl.sum(product, axis=1) * scale  # (BLOCK_SIZE_M, BLOCK_SIZE_N)
        
        mask_qk = mask_m[:, None] & kv_mask[None, :]
        qk = tl.where(mask_qk, qk, float("-inf"))
        
        # Online softmax update
        # Find max only over valid (non-masked) positions
        qk_masked = tl.where(mask_qk, qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk_masked, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        # Note: masked positions have qk=-inf, so p=0 for them automatically
        # Sum over all positions (masked positions contribute 0)
        l_ij = tl.sum(p, axis=1)
        
        # Load V block: (BLOCK_SIZE_N, HEAD_DIM)
        v_dim = tl.arange(0, HEAD_DIM)
        v_mask = v_dim < dv
        v_offsets = v_head_off + kv_idx[:, None] * stride_vm + v_dim[None, :] * stride_vd
        v_mask_full = kv_mask[:, None] & v_mask[None, :]
        v = tl.load(v_ptr + v_offsets, mask=v_mask_full, other=0.0).to(tl.float32)
        
        # Compute p @ v: (BLOCK_SIZE_M, BLOCK_SIZE_N) @ (BLOCK_SIZE_N, HEAD_DIM)
        # Use tl.dot for better precision when dimensions are large enough
        # For decode, BLOCK_SIZE_M=1, so we need to check all dimensions
        if USE_DOT and dv >= 16 and BLOCK_SIZE_M >= 16:
            # Use tl.dot with IEEE precision
            pv = tl.dot(p, v, input_precision="ieee")  # (BLOCK_SIZE_M, HEAD_DIM)
        else:
            # Manual matrix multiplication for small dimensions
            p_expanded = p[:, :, None]  # (BLOCK_SIZE_M, BLOCK_SIZE_N, 1)
            v_expanded = v[None, :, :]  # (1, BLOCK_SIZE_N, HEAD_DIM)
            pv_product = p_expanded * v_expanded  # (BLOCK_SIZE_M, BLOCK_SIZE_N, HEAD_DIM)
            # Apply mask to product before summing
            mask_qk_expanded = mask_qk[:, :, None]  # (BLOCK_SIZE_M, BLOCK_SIZE_N, 1)
            pv_product = tl.where(mask_qk_expanded, pv_product, 0.0)
            pv = tl.sum(pv_product, axis=1)  # (BLOCK_SIZE_M, HEAD_DIM)
        
        # Rescale previous accumulator and add new contribution
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None] + pv
        
        # Update running statistics
        m_i = m_ij
        l_i = l_i * alpha + l_ij
    
    # Final normalization
    acc = acc / l_i[:, None]
    
    # Store output
    o_dim = tl.arange(0, HEAD_DIM)
    o_mask = o_dim < dv
    o_offsets = out_head_off + offs_m[:, None] * stride_om + o_dim[None, :] * stride_od
    o_mask_full = mask_m[:, None] & o_mask[None, :]
    tl.store(attn_val_ptr + o_offsets, acc.to(OUT_DTYPE), mask=o_mask_full)


def kernel(q, k, v, attn_val, scale, BLOCK_SIZE=8):
    """Kernel entry point for self-attention.
    
    Automatically selects prefill or decode kernel based on seq_len.
    
    Args:
        q: torch.Tensor (seq_len, n_head, d)
        k: torch.Tensor (total_len, nkv_head, d)
        v: torch.Tensor (total_len, nkv_head, dv)
        attn_val: torch.Tensor (seq_len, n_head, dv)
        scale: float scaling factor
        BLOCK_SIZE: Block size for KV positions
    """

    seq_len, n_head, d = q.shape
    total_len, nkv_head, _ = k.shape
    _, _, dv = v.shape

    # HEAD_DIM must be a power of 2 for efficient processing
    HEAD_DIM = triton.next_power_of_2(max(d, dv))
    
    # Block sizes
    BLOCK_SIZE_M = min(32, triton.next_power_of_2(seq_len))
    BLOCK_SIZE_N = BLOCK_SIZE
    
    USE_DOT = d >= 16 and BLOCK_SIZE_M >= 16 and BLOCK_SIZE_N >= 16

    if attn_val.dtype == torch.float32:
        OUT_DTYPE = tl.float32
    elif attn_val.dtype == torch.float16:
        OUT_DTYPE = tl.float16
    elif attn_val.dtype == torch.bfloat16:
        OUT_DTYPE = tl.bfloat16
    else:
        raise TypeError(f"Unsupported dtype: {attn_val.dtype}")

    # Grid: (num_query_blocks, n_heads)
    grid = (triton.cdiv(seq_len, BLOCK_SIZE_M), n_head)    # Select kernel based on seq_len
    if seq_len == 1:
        # Decode stage: single query position
        self_attention_decode_kernel[grid](
            q, k, v, attn_val,
            seq_len, total_len, n_head, nkv_head, d, dv, scale,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            attn_val.stride(0), attn_val.stride(1), attn_val.stride(2),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            HEAD_DIM=HEAD_DIM,
            USE_DOT=USE_DOT,
            OUT_DTYPE=OUT_DTYPE
        )
    else:
        # Prefill stage: multiple query positions
        self_attention_prefill_kernel[grid](
            q, k, v, attn_val,
            seq_len, total_len, n_head, nkv_head, d, dv, scale,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            attn_val.stride(0), attn_val.stride(1), attn_val.stride(2),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            HEAD_DIM=HEAD_DIM,
            USE_DOT=USE_DOT,
            OUT_DTYPE=OUT_DTYPE
        )
