# 大模型计算 HW3 - llaisys 推理系统

## 项目简介

本项目实现了一个高性能的大语言模型推理系统 **llaisys**，支持 Qwen2 系列模型的推理。系统采用 C++ 后端和 Triton GPU kernels 优化，提供高效的模型推理能力。

## 主要特性

- ✅ **Qwen2 模型支持**：完整实现 Qwen2 模型的推理流程
- ✅ **多设备支持**：支持 CPU 和 NVIDIA GPU 推理
- ✅ **高性能优化**：使用 Triton 编写优化的 GPU kernels
- ✅ **完整算子实现**：包含自注意力、SwiGLU、RoPE、RMSNorm 等核心算子
- ✅ **量化支持**：支持 INT8 和 Float8 量化推理
- ✅ **KV Cache**：实现高效的 KV Cache 机制

## 项目结构

```
llaisys-thu/
├── python/                    # Python 接口层
│   └── llaisys/
│       ├── models/            # 模型实现（Qwen2）
│       ├── libllaisys/        # C++ 库的 Python 绑定
│       │   └── triton/        # Triton GPU kernels
│       │       └── kernels/   # 各种优化算子
│       ├── tensor.py          # 张量操作
│       ├── ops.py             # 算子接口
│       └── runtime.py         # 运行时 API
├── src/                       # C++ 源代码
│   ├── core/                 # 核心功能
│   ├── device/               # 设备抽象层
│   ├── ops/                  # 算子实现
│   └── tensor/               # 张量实现
├── include/                   # C++ 头文件
├── test/                      # 测试代码
│   ├── test_infer.py         # 推理测试
│   └── ops/                  # 算子单元测试
├── scripts/                   # 构建脚本
└── build/                     # 构建输出目录
```

## 已实现的算子

系统实现了以下算子（使用 Triton 优化）：

- **add**: 张量加法
- **argmax**: 最大值索引
- **embedding**: 词嵌入查找
- **linear**: 线性变换（矩阵乘法）
- **rms_norm**: RMS 归一化
- **rope**: 旋转位置编码（RoPE）
- **self_attention**: 自注意力机制
- **swiglu**: SwiGLU 激活函数

## 环境要求

### 系统要求
- Linux 系统
- CUDA 12.8 (GPU 推理需要)
- xmake 构建工具

### Python 依赖
```bash
pip install torch transformers huggingface_hub safetensors
```

## 编译安装

### 1. 安装 xmake（如果未安装）

```bash
# 使用官方安装脚本
bash <(curl -s https://xmake.io/shget.text)
```

### 2. 编译项目

```bash
cd llaisys-thu

# 仅编译 CPU 版本
xmake

# 编译包含 NVIDIA GPU 支持的版本
xmake config --nv-gpu=y
xmake
```

### 3. 安装 Python 包

```bash
cd python
pip install -e .
```

## 使用方法

### 推理测试

运行完整的推理测试，对比 HuggingFace 实现和 llaisys 实现：

```bash
# 使用 GPU 推理
ENABLE_TRITON=True python test/test_infer.py --test --device nvidia

# 使用 CPU 推理
python test/test_infer.py --test --device cpu

# 自定义参数
python test/test_infer.py \
    --device nvidia \
    --model /path/to/model \
    --prompt "你的问题" \
    --max_steps 128 \
    --top_p 0.8 \
    --top_k 50 \
    --temperature 0.8
```

### 算子测试

测试单个算子的正确性和性能：

```bash
# 测试自注意力算子
ENABLE_TRITON=True python test/ops/self_attention.py --device nvidia

# 测试其他算子
ENABLE_TRITON=True python test/ops/swiglu.py --device nvidia
ENABLE_TRITON=True python test/ops/rope.py --device nvidia
ENABLE_TRITON=True python test/ops/rms_norm.py --device nvidia
```

### 使用脚本

项目提供了便捷的测试脚本：

```bash
# 推理测试脚本
bash test_infer.sh

# 算子测试脚本
bash test_ops.sh
```

## 量化推理（由于该部分效果不佳，已经被删除）

系统支持量化推理以降低内存占用和提高推理速度：

```bash
# INT8 量化
python test/test_infer.py --device nvidia --quant int8

# Float8 量化
python test/test_infer.py --device nvidia --quant float8
```

## 性能分析

使用 Nsight Systems 进行性能分析：

```bash
ENABLE_TRITON=True \
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  -o test_python_profile \
  python test/test_infer.py --test --device nvidia
```

## 代码示例

### 基本使用

```python
import llaisys
from transformers import AutoTokenizer

# 加载模型
model = llaisys.models.Qwen2(
    model_path="/path/to/model",
    device=llaisys.DeviceType.NVIDIA
)

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("/path/to/model")

# 准备输入
prompt = "你好，请介绍一下自己。"
input_content = tokenizer.apply_chat_template(
    conversation=[{"role": "user", "content": prompt}],
    add_generation_prompt=True,
    tokenize=False,
)
inputs = tokenizer.encode(input_content)

# 生成
outputs = model.generate(
    inputs,
    max_new_tokens=128,
    top_k=50,
    top_p=0.8,
    temperature=0.8
)

# 解码结果
result = tokenizer.decode(outputs, skip_special_tokens=True)
print(result)
```

## 开发说明

### 添加新算子

1. 在 `python/llaisys/libllaisys/triton/kernels/` 下创建 Triton kernel
2. 在 `src/ops/` 下实现 C++ 算子
3. 在 `python/llaisys/ops.py` 中添加 Python 接口
4. 在 `test/ops/` 下添加测试用例

### 调试模式

使用调试模式编译以获取更详细的错误信息：

```bash
xmake f -m debug
xmake
```

## 测试

运行所有测试：

```bash
# 推理正确性测试
python test/test_infer.py --test --device nvidia

# 算子测试
python test/ops/self_attention.py --device nvidia
python test/ops/swiglu.py --device nvidia
# ... 其他算子测试
```

## 常见问题

### 1. Triton 未启用

确保设置环境变量：
```bash
export ENABLE_TRITON=True
```

### 2. CUDA 相关错误

- 检查 CUDA 版本是否兼容
- 确认 GPU 驱动已正确安装
- 验证 CUDA 环境变量设置

### 3. 模型加载失败

- 确认模型路径正确
- 检查模型文件完整性
- 验证 safetensors 格式支持
