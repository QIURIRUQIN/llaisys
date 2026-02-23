"""Triton kernel for argmax operation - works with llaisys tensors."""

import triton
import triton.language as tl

@triton.jit
def argmax_kernel_single(
    max_idx_ptr,
    max_val_ptr,
    vals_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Find argmax for tensors that fit in a single block."""
    # Process all elements in one block
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values
    vals = tl.load(vals_ptr + offsets, mask=mask, other=float('-inf'))
    
    # Find max value
    local_max = tl.max(vals, axis=0)
    
    # Find index of max value (first occurrence)
    is_max = (vals == local_max) & mask # vals == local_max是一次广播比较，返回一个bool mask，在等于local_mask的offsets位置为true
    local_idx = tl.where(is_max, offsets, n_elements)
    local_idx = tl.min(local_idx, axis=0)
    
    # Store results
    tl.store(max_val_ptr, local_max)
    tl.store(max_idx_ptr, local_idx)


@triton.jit
def argmax_kernel_large(
    max_idx_ptr,
    max_val_ptr,
    vals_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Find argmax for large tensors using iterative approach within kernel."""
    # Initialize with first element
    best_val = tl.load(vals_ptr).to(tl.float32) # 全局最大变量，由于需要与max操作后的float32比较，所以需要转换
    best_idx = 0
    
    # Process all elements in chunks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load values for this block
        vals = tl.load(vals_ptr + offsets, mask=mask, other=float('-inf'))
        
        # Find local max in this block
        local_max = tl.max(vals, axis=0) # 这里triton的reduce操作会自动返回float32
        
        # Update global max if this block has larger value
        if local_max > best_val:
            # Find index within block
            is_max = (vals == local_max) & mask
            local_idx = tl.where(is_max, offsets, n_elements)
            local_idx = tl.min(local_idx, axis=0)
            
            best_val = local_max
            best_idx = local_idx
    
    # Store results
    tl.store(max_val_ptr, best_val) #存储回去的时候会自动转换会原本的dtype
    tl.store(max_idx_ptr, best_idx)


def kernel(vals, max_idx, max_val):
    """Compute argmax using Triton kernel with llaisys tensors.
    
    Args:
        vals: torch.Tensor input values (converted from llaisys tensor)
        max_idx: torch.Tensor output index (single element, int64)
        max_val: torch.Tensor output value (single element)
    """
    n_elements = vals.numel()
    
    # Use BLOCK_SIZE of 1024 for iterative processing
    BLOCK_SIZE = min(1024, triton.next_power_of_2(n_elements))
    
    # For tensors that fit in one block, use optimized single kernel
    if n_elements <= BLOCK_SIZE:
        argmax_kernel_single[(1,)](
            max_idx,
            max_val,
            vals,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # For larger tensors, use iterative kernel
        argmax_kernel_large[(1,)](
            max_idx,
            max_val,
            vals,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
