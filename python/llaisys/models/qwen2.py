from typing import Sequence, Dict, List, Optional
from ..libllaisys import DeviceType, DataType, MemcpyKind
from ..tensor import Tensor
from ..ops import Ops
from ..runtime import RuntimeAPI

from pathlib import Path
import json


class Qwen2:
    """Qwen2 model implementation using llaisys custom operators."""

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        self.device = device
        self.runtime = RuntimeAPI(device)
        if device == DeviceType.NVIDIA:
            self.runtime.set_device(0)
        
        model_path = Path(model_path)
        
        # Load config
        with open(model_path / "config.json") as f:
            config = json.load(f)
        
        self.hidden_size = config["hidden_size"]
        self.intermediate_size = config["intermediate_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.num_key_value_heads = config["num_key_value_heads"]
        self.num_hidden_layers = config["num_hidden_layers"]
        self.vocab_size = config["vocab_size"]
        self.rms_norm_eps = config["rms_norm_eps"]
        self.rope_theta = config["rope_theta"]
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.max_position_embeddings = config.get("max_position_embeddings", 131072)
        
        # Use bfloat16 for weights
        self.dtype = DataType.BF16
        
        # Initialize weight storage
        self.weights: Dict[str, Tensor] = {}
        
        # Load weights from safetensors
        self._load_weights(model_path)
        
        # KV Cache
        self.kv_cache: Dict[int, Dict[str, Tensor]] = {}
        self.kv_cache_pos = 0
        self._init_kv_cache()
        
        # Reusable tensors for efficiency
        self._pos_ids_cache: Optional[Tensor] = None
        self._token_buf: Optional[Tensor] = None
        self._token_buf_cpu = None
        self._argmax_idx: Optional[Tensor] = None
        self._argmax_val: Optional[Tensor] = None
        self._argmax_cpu = None
    
    def _load_weights(self, model_path: Path):
        """Load model weights from safetensors files."""
        from safetensors.torch import load_file
        import torch
        
        weights_torch = {}
        for file in sorted(model_path.glob("*.safetensors")):
            weights_torch.update(load_file(str(file)))
        
        for name, tensor in weights_torch.items():
            self.weights[name] = self._torch_to_llaisys(tensor)
    
    def _torch_to_llaisys(self, torch_tensor) -> Tensor:
        """Convert a torch tensor to llaisys Tensor on the target device."""
        import torch
        
        # Ensure contiguous
        torch_tensor = torch_tensor.contiguous()
        
        # Map torch dtype to llaisys dtype
        dtype_map = {
            torch.float32: DataType.F32,
            torch.float16: DataType.F16,
            torch.bfloat16: DataType.BF16,
            torch.int64: DataType.I64,
            torch.int32: DataType.I32,
        }
        llaisys_dtype = dtype_map.get(torch_tensor.dtype, DataType.F32)
        
        # Create llaisys tensor
        shape = tuple(torch_tensor.shape)
        llaisys_tensor = Tensor(
            shape=shape,
            dtype=llaisys_dtype,
            device=self.device
        )
        
        # Copy data
        byte_size = torch_tensor.numel() * torch_tensor.element_size()
        if self.device == DeviceType.CPU:
            self.runtime.memcpy_sync(
                llaisys_tensor.data_ptr(),
                torch_tensor.data_ptr(),
                byte_size,
                MemcpyKind.H2H
            )
        else:
            self.runtime.memcpy_sync(
                llaisys_tensor.data_ptr(),
                torch_tensor.data_ptr(),
                byte_size,
                MemcpyKind.H2D
            )
        
        return llaisys_tensor
    
    def _init_kv_cache(self):
        """Initialize KV cache for all layers."""
        for layer_idx in range(self.num_hidden_layers):
            k_cache = Tensor(
                shape=(self.max_position_embeddings, self.num_key_value_heads, self.head_dim),
                dtype=self.dtype,
                device=self.device
            )
            v_cache = Tensor(
                shape=(self.max_position_embeddings, self.num_key_value_heads, self.head_dim),
                dtype=self.dtype,
                device=self.device
            )
            self.kv_cache[layer_idx] = {"k": k_cache, "v": v_cache}
    
    def _create_tensor(self, shape, dtype=None) -> Tensor:
        """Create a tensor on the target device."""
        if dtype is None:
            dtype = self.dtype
        return Tensor(shape=shape, dtype=dtype, device=self.device)
    
    def _get_weight(self, name: str, optional: bool = False) -> Optional[Tensor]:
        """Get weight tensor by name."""
        if name not in self.weights:
            if optional:
                return None
            raise KeyError(f"Weight {name} not found")
        return self.weights[name]
    
    def _create_pos_ids(self, start_pos: int, seq_len: int) -> Tensor:
        """Create position IDs tensor."""
        import torch
        pos_ids = torch.arange(start_pos, start_pos + seq_len, dtype=torch.int64)
        return self._torch_to_llaisys(pos_ids)
    
    def _forward_embedding(self, input_ids: Tensor) -> Tensor:
        """Forward pass through embedding layer."""
        seq_len = input_ids.shape()[0]
        hidden_states = self._create_tensor((seq_len, self.hidden_size))
        Ops.embedding(hidden_states, input_ids, self.weights["model.embed_tokens.weight"])
        return hidden_states
    
    def _forward_rms_norm(self, x: Tensor, weight_name: str) -> Tensor:
        """Forward pass through RMS normalization."""
        out = self._create_tensor(x.shape())
        Ops.rms_norm(out, x, self.weights[weight_name], self.rms_norm_eps)
        return out
    
    def _forward_attention(self, hidden_states: Tensor, layer_idx: int,
                           pos_ids: Tensor, start_pos: int, is_prefill: bool) -> Tensor:
        """Self-attention with KV cache."""
        seq_len = hidden_states.shape()[0]
        kv_dim = self.num_key_value_heads * self.head_dim
        
        # Get weights
        q_weight = self._get_weight(f"model.layers.{layer_idx}.self_attn.q_proj.weight")
        k_weight = self._get_weight(f"model.layers.{layer_idx}.self_attn.k_proj.weight")
        v_weight = self._get_weight(f"model.layers.{layer_idx}.self_attn.v_proj.weight")
        o_weight = self._get_weight(f"model.layers.{layer_idx}.self_attn.o_proj.weight")
        
        # Optional biases
        q_bias = self._get_weight(f"model.layers.{layer_idx}.self_attn.q_proj.bias", optional=True)
        k_bias = self._get_weight(f"model.layers.{layer_idx}.self_attn.k_proj.bias", optional=True)
        v_bias = self._get_weight(f"model.layers.{layer_idx}.self_attn.v_proj.bias", optional=True)
        
        # Q projection: output shape (seq_len, hidden_size), then view as (seq_len, num_heads, head_dim)
        q_out = self._create_tensor((seq_len, self.hidden_size))
        Ops.linear(q_out, hidden_states, q_weight, q_bias)
        q = q_out.view(seq_len, self.num_attention_heads, self.head_dim)
        
        # Apply RoPE to Q
        Ops.rope(q, q, pos_ids, self.rope_theta)
        
        # Get KV cache write views
        k_cache = self.kv_cache[layer_idx]["k"]
        v_cache = self.kv_cache[layer_idx]["v"]
        k_dst = k_cache.slice(0, start_pos, start_pos + seq_len)
        v_dst = v_cache.slice(0, start_pos, start_pos + seq_len)
        
        # K projection: write directly to cache
        Ops.linear(k_dst.view(seq_len, kv_dim), hidden_states, k_weight, k_bias)
        # Apply RoPE to K
        Ops.rope(k_dst, k_dst, pos_ids, self.rope_theta)
        
        # V projection: write directly to cache
        Ops.linear(v_dst.view(seq_len, kv_dim), hidden_states, v_weight, v_bias)
        
        # Get full KV from cache
        kv_len = start_pos + seq_len
        k_full = k_cache.slice(0, 0, kv_len)
        v_full = v_cache.slice(0, 0, kv_len)
        
        # Compute attention
        attn_output = self._create_tensor((seq_len, self.num_attention_heads, self.head_dim))
        scale = 1.0 / (self.head_dim ** 0.5)
        Ops.self_attention(attn_output, q, k_full, v_full, scale)
        
        # Output projection
        output = self._create_tensor((seq_len, self.hidden_size))
        Ops.linear(output, attn_output.view(seq_len, self.hidden_size), o_weight, None)
        
        return output
    
    def _forward_mlp(self, hidden_states: Tensor, layer_idx: int) -> Tensor:
        """MLP with SwiGLU activation."""
        seq_len = hidden_states.shape()[0]
        
        gate_weight = self._get_weight(f"model.layers.{layer_idx}.mlp.gate_proj.weight")
        up_weight = self._get_weight(f"model.layers.{layer_idx}.mlp.up_proj.weight")
        down_weight = self._get_weight(f"model.layers.{layer_idx}.mlp.down_proj.weight")
        
        gate = self._create_tensor((seq_len, self.intermediate_size))
        up = self._create_tensor((seq_len, self.intermediate_size))
        Ops.linear(gate, hidden_states, gate_weight, None)
        Ops.linear(up, hidden_states, up_weight, None)
        
        swiglu_out = self._create_tensor((seq_len, self.intermediate_size))
        Ops.swiglu(swiglu_out, gate, up)
        
        output = self._create_tensor((seq_len, self.hidden_size))
        Ops.linear(output, swiglu_out, down_weight, None)
        
        return output
    
    def _forward_layer(self, hidden_states: Tensor, layer_idx: int,
                       pos_ids: Tensor, start_pos: int, is_prefill: bool) -> Tensor:
        """Forward pass through a single transformer layer."""
        seq_len = hidden_states.shape()[0]
        
        # Input LayerNorm
        normed = self._forward_rms_norm(hidden_states, f"model.layers.{layer_idx}.input_layernorm.weight")
        
        # Self-attention
        attn_output = self._forward_attention(normed, layer_idx, pos_ids, start_pos, is_prefill)
        
        # Residual connection
        residual1 = self._create_tensor((seq_len, self.hidden_size))
        Ops.add(residual1, hidden_states, attn_output)
        
        # Post-attention LayerNorm
        normed2 = self._forward_rms_norm(residual1, f"model.layers.{layer_idx}.post_attention_layernorm.weight")
        
        # MLP
        mlp_output = self._forward_mlp(normed2, layer_idx)
        
        # Residual connection
        output = self._create_tensor((seq_len, self.hidden_size))
        Ops.add(output, residual1, mlp_output)
        
        return output
    
    def _forward(self, input_ids: Tensor, start_pos: int, is_prefill: bool) -> Tensor:
        """Forward pass through the model."""
        seq_len = input_ids.shape()[0]
        
        # Embedding
        hidden_states = self._forward_embedding(input_ids)
        
        # Position IDs
        pos_ids = self._create_pos_ids(start_pos, seq_len)
        
        # Transformer layers
        for layer_idx in range(self.num_hidden_layers):
            hidden_states = self._forward_layer(hidden_states, layer_idx, pos_ids, start_pos, is_prefill)
        
        # Final LayerNorm
        normed = self._forward_rms_norm(hidden_states, "model.norm.weight")
        
        # LM head
        logits = self._create_tensor((seq_len, self.vocab_size))
        Ops.linear(logits, normed, self.weights["lm_head.weight"], None)
        
        return logits
    
    def _sample_argmax(self, logits: Tensor) -> int:
        """Sample using argmax (greedy decoding)."""
        vocab_size = logits.shape()[0]
        import torch
        
        if self._argmax_idx is None:
            self._argmax_idx = self._create_tensor((1,), DataType.I64)
            self._argmax_val = self._create_tensor((1,), self.dtype)
            self._argmax_cpu = torch.zeros(1, dtype=torch.int64)
        
        Ops.argmax(self._argmax_idx, self._argmax_val, logits)
        
        # Copy result back to CPU
        if self.device == DeviceType.CPU:
            self.runtime.memcpy_sync(
                self._argmax_cpu.data_ptr(),
                self._argmax_idx.data_ptr(),
                8,
                MemcpyKind.H2H
            )
        else:
            self.runtime.memcpy_sync(
                self._argmax_cpu.data_ptr(),
                self._argmax_idx.data_ptr(),
                8,
                MemcpyKind.D2H
            )
        
        return int(self._argmax_cpu[0].item())

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 128,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ) -> List[int]:
        """Generate tokens autoregressively."""
        import torch
        
        if len(inputs) == 0:
            return []
        
        # Reset KV cache position
        self.kv_cache_pos = 0
        
        # Convert input to tensor (use I64 for embedding lookup)
        input_tensor = torch.tensor(inputs, dtype=torch.int64)
        input_ids = self._torch_to_llaisys(input_tensor)
        
        # Prefill: process all input tokens at once
        logits = self._forward(input_ids, start_pos=0, is_prefill=True)
        self.kv_cache_pos = len(inputs)
        
        # Get logits for the last position
        last_logits = logits.slice(0, len(inputs) - 1, len(inputs))
        last_logits = last_logits.view(self.vocab_size)
        
        # Sample next token
        next_token = self._sample_argmax(last_logits)
        
        # Output tokens (including input)
        output_tokens = list(inputs) + [next_token]
        
        # Decode: generate one token at a time
        for i in range(max_new_tokens - 1):
            # Check for EOS
            if next_token == 151643:
                break
            
            # Create input tensor for single token
            single_input = torch.tensor([next_token], dtype=torch.int64)
            single_input_ids = self._torch_to_llaisys(single_input)
            
            # Forward pass
            start_pos = self.kv_cache_pos
            logits = self._forward(single_input_ids, start_pos=start_pos, is_prefill=False)
            self.kv_cache_pos += 1
            
            # Get logits (only 1 position)
            logits = logits.view(self.vocab_size)
            
            # Sample next token
            next_token = self._sample_argmax(logits)
            output_tokens.append(next_token)
        
        return output_tokens
