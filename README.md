# GPU 显存计算器

根据模型参数，计算推理和训练所需的 GPU 显存。

## 功能

- 支持 20+ 常见开源模型预设（LLaMA、Qwen、Mistral、DeepSeek 等）
- 支持多种精度：FP32、FP16/BF16、INT8、INT4
- 推理模式：计算模型参数 + KV Cache + 激活值
- 训练模式：额外计算梯度 + 优化器状态 + 激活值（支持梯度检查点）
- 自动推荐 GPU 配置
- 支持 JSON 输出、交互模式

## 使用方法

```bash
# 计算 LLaMA-7B FP16 推理显存
python gpu_memory_calculator.py --model llama-7b --precision fp16

# 计算 70B 模型 INT4 量化推理
python gpu_memory_calculator.py --model llama2-70b --precision int4 --seq-length 4096

# 训练模式
python gpu_memory_calculator.py --model llama2-7b --precision bf16 --mode train \
    --batch-size 4 --optimizer adam --gradient-checkpointing

# 自定义模型参数
python gpu_memory_calculator.py --params 13 --hidden 5120 --layers 40 --heads 40 \
    --precision bf16 --mode train

# JSON 输出
python gpu_memory_calculator.py --model llama-7b --precision fp16 --json

# 列出预设模型 / GPU 规格
python gpu_memory_calculator.py --list-models
python gpu_memory_calculator.py --list-gpus

# 交互模式
python gpu_memory_calculator.py --interactive
```

## 计算原理

| 组件 | 计算方式 |
|------|----------|
| 模型参数 | 参数量 × 每参数字节数 |
| KV Cache | 2 × 层数 × 序列长度 × KV头数 × 头维度 × 字节数 × batch |
| 激活值 | 12 × hidden × seq_len × batch × 字节数 × 层数 |
| 梯度 | 与模型参数相同 |
| 优化器 (Adam) | 参数量 × 8 字节 (一阶矩 + 二阶矩) |
