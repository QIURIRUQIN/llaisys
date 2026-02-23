"""Utility functions for Triton kernels with llaisys tensors."""

import ctypes
from ...llaisys_types import DataType, DeviceType, MemcpyKind


# Element sizes for each dtype
DTYPE_SIZES = {
    DataType.F16: 2,
    DataType.F32: 4,
    DataType.F64: 8,
    DataType.BF16: 2,
    DataType.I32: 4,
    DataType.I64: 8,
    DataType.U32: 4,
    DataType.U64: 8,
    DataType.BOOL: 1,
    DataType.I8: 1,
    DataType.U8: 1,
    DataType.I16: 2,
    DataType.U16: 2,
}


def get_element_size(dtype: DataType) -> int:
    """Get element size in bytes for a dtype."""
    return DTYPE_SIZES.get(dtype, 4)


def ptr_to_int(ptr) -> int:
    """Convert ctypes pointer to integer for Triton."""
    if ptr is None:
        return 0
    if isinstance(ptr, ctypes.c_void_p):
        return ptr.value if ptr.value is not None else 0
    if hasattr(ptr, 'value'):
        return ptr.value if ptr.value is not None else 0
    return int(ptr)


def dtype_to_str(dtype: DataType) -> str:
    """Convert llaisys DataType to string."""
    dtype_map = {
        DataType.F32: "f32",
        DataType.F16: "f16",
        DataType.BF16: "bf16",
        DataType.I32: "i32",
        DataType.I64: "i64",
    }
    return dtype_map.get(dtype, "f32")


class TritonTensorWrapper:
    """
    Wrapper to make llaisys tensors or torch tensors compatible with Triton.
    
    Triton needs objects with data_ptr() and dtype attributes.
    This wrapper provides those using the underlying tensor.
    """
    
    def __init__(self, tensor):
        """
        Args:
            tensor: llaisys.Tensor object or torch.Tensor object
        """
        self._tensor = tensor
        self._ptr = ptr_to_int(tensor.data_ptr())
        
        # Try to access dtype as attribute (torch tensor) first
        # If that fails or returns a callable, try as method (llaisys tensor)
        try:
            dtype_attr = getattr(tensor, 'dtype', None)
            if dtype_attr is not None and not callable(dtype_attr):
                # It's a torch tensor: dtype is a property/attribute
                self._dtype = tensor.dtype
                self._shape = tensor.shape
                # torch tensor: stride() is a method, but we need strides tuple
                if hasattr(tensor, 'stride') and callable(tensor.stride):
                    self._strides = tensor.stride()
                else:
                    self._strides = getattr(tensor, 'strides', ())
                self._is_torch = True
            else:
                # It's likely a llaisys tensor: dtype() is a method
                raise AttributeError("Try llaisys tensor")
        except (AttributeError, TypeError):
            # Handle llaisys tensor
            self._dtype = tensor.dtype()  # llaisys tensor: dtype() is method
            self._shape = tensor.shape()  # llaisys tensor: shape() is method
            self._strides = tensor.strides()  # llaisys tensor: strides() is method
            self._is_torch = False
    
    def data_ptr(self) -> int:
        """Return raw pointer as integer."""
        return self._ptr
    
    @property
    def dtype(self):
        """Return a dtype object compatible with Triton."""
        # If it's already a torch tensor, return its dtype directly
        if self._is_torch:
            return self._dtype
        
        # For llaisys tensor, convert DataType to torch dtype
        # Triton uses torch dtypes internally, so we need to import torch
        # only for dtype mapping, not for tensor operations
        try:
            import torch
            dtype_map = {
                DataType.F32: torch.float32,
                DataType.F16: torch.float16,
                DataType.BF16: torch.bfloat16,
                DataType.I32: torch.int32,
                DataType.I64: torch.int64,
            }
            return dtype_map.get(self._dtype, torch.float32)
        except ImportError:
            return None
    
    @property
    def shape(self):
        return self._shape
    
    @property 
    def strides(self):
        return self._strides
    
    def numel(self) -> int:
        """Return total number of elements."""
        if self._is_torch:
            return self._tensor.numel()
        n = 1
        for s in self._shape:
            n *= s
        return n
    
    def element_size(self) -> int:
        """Return element size in bytes."""
        if self._is_torch:
            # For torch tensor, use its element_size() method
            return self._tensor.element_size()
        return get_element_size(self._dtype)


def wrap_tensor(tensor) -> TritonTensorWrapper:
    """Wrap an llaisys tensor or torch tensor for Triton compatibility."""
    return TritonTensorWrapper(tensor)

