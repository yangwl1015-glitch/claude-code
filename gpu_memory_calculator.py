#!/usr/bin/env python3
"""
GPU Memory Calculator - 根据模型参数计算所需显存

支持功能：
- 根据模型参数量计算推理/训练所需显存
- 支持不同精度（FP32, FP16/BF16, INT8, INT4）
- 支持常见开源模型的预设参数
- 计算训练时的额外显存开销（优化器状态、梯度、激活值等）
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Precision(Enum):
    FP32 = ("FP32", 4)
    FP16 = ("FP16", 2)
    BF16 = ("BF16", 2)
    INT8 = ("INT8", 1)
    INT4 = ("INT4", 0.5)

    def __init__(self, label: str, bytes_per_param: float):
        self.label = label
        self.bytes_per_param = bytes_per_param


# 常见模型预设参数（参数量单位：十亿/Billion）
PRESET_MODELS = {
    # LLaMA 系列
    "llama-7b":   {"params_b": 7,    "hidden": 4096,  "layers": 32,  "heads": 32,  "vocab": 32000},
    "llama-13b":  {"params_b": 13,   "hidden": 5120,  "layers": 40,  "heads": 40,  "vocab": 32000},
    "llama-33b":  {"params_b": 33,   "hidden": 6656,  "layers": 60,  "heads": 52,  "vocab": 32000},
    "llama-65b":  {"params_b": 65,   "hidden": 8192,  "layers": 80,  "heads": 64,  "vocab": 32000},
    # LLaMA 2
    "llama2-7b":  {"params_b": 7,    "hidden": 4096,  "layers": 32,  "heads": 32,  "vocab": 32000},
    "llama2-13b": {"params_b": 13,   "hidden": 5120,  "layers": 40,  "heads": 40,  "vocab": 32000},
    "llama2-70b": {"params_b": 70,   "hidden": 8192,  "layers": 80,  "heads": 64,  "vocab": 32000},
    # LLaMA 3
    "llama3-8b":  {"params_b": 8,    "hidden": 4096,  "layers": 32,  "heads": 32,  "vocab": 128256},
    "llama3-70b": {"params_b": 70.6, "hidden": 8192,  "layers": 80,  "heads": 64,  "vocab": 128256},
    # Qwen 系列
    "qwen-7b":    {"params_b": 7,    "hidden": 4096,  "layers": 32,  "heads": 32,  "vocab": 151936},
    "qwen-14b":   {"params_b": 14,   "hidden": 5120,  "layers": 40,  "heads": 40,  "vocab": 151936},
    "qwen-72b":   {"params_b": 72,   "hidden": 8192,  "layers": 80,  "heads": 64,  "vocab": 151936},
    "qwen2.5-7b": {"params_b": 7,    "hidden": 3584,  "layers": 28,  "heads": 28,  "vocab": 151936},
    "qwen2.5-72b":{"params_b": 72,   "hidden": 8192,  "layers": 80,  "heads": 64,  "vocab": 151936},
    # Mistral / Mixtral
    "mistral-7b":   {"params_b": 7.3,  "hidden": 4096,  "layers": 32,  "heads": 32,  "vocab": 32000},
    "mixtral-8x7b": {"params_b": 46.7, "hidden": 4096,  "layers": 32,  "heads": 32,  "vocab": 32000},
    # ChatGLM
    "chatglm3-6b":  {"params_b": 6.2,  "hidden": 4096,  "layers": 28,  "heads": 32,  "vocab": 65024},
    # DeepSeek
    "deepseek-7b":  {"params_b": 6.9,  "hidden": 4096,  "layers": 30,  "heads": 32,  "vocab": 102400},
    "deepseek-67b": {"params_b": 67,   "hidden": 8192,  "layers": 95,  "heads": 64,  "vocab": 102400},
    # GPT 系列（估计值）
    "gpt2":       {"params_b": 0.117, "hidden": 768,   "layers": 12,  "heads": 12,  "vocab": 50257},
    "gpt2-medium":{"params_b": 0.345, "hidden": 1024,  "layers": 24,  "heads": 16,  "vocab": 50257},
    "gpt2-large": {"params_b": 0.774, "hidden": 1280,  "layers": 36,  "heads": 20,  "vocab": 50257},
    "gpt2-xl":    {"params_b": 1.5,   "hidden": 1600,  "layers": 48,  "heads": 25,  "vocab": 50257},
}


@dataclass
class MemoryEstimate:
    """显存估算结果"""
    model_params_gb: float       # 模型参数占用 (GB)
    kv_cache_gb: float           # KV Cache 占用 (GB)
    activation_gb: float         # 激活值占用 (GB)
    optimizer_gb: float          # 优化器状态占用 (GB)
    gradient_gb: float           # 梯度占用 (GB)
    total_gb: float              # 总计 (GB)
    description: str             # 描述


def format_bytes(gb: float) -> str:
    """格式化显存大小"""
    if gb >= 1:
        return f"{gb:.2f} GB"
    return f"{gb * 1024:.1f} MB"


def calc_model_memory(params_b: float, precision: Precision) -> float:
    """计算模型参数占用的显存 (GB)"""
    total_bytes = params_b * 1e9 * precision.bytes_per_param
    return total_bytes / (1024 ** 3)


def calc_kv_cache(
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    batch_size: int,
    seq_length: int,
    precision: Precision,
    num_kv_heads: Optional[int] = None,
) -> float:
    """
    计算 KV Cache 占用的显存 (GB)
    KV Cache = 2 * num_layers * seq_len * num_kv_heads * head_dim * bytes_per_param * batch_size
    """
    if num_kv_heads is None:
        num_kv_heads = num_heads
    head_dim = hidden_size // num_heads
    kv_cache_bytes = (
        2 * num_layers * seq_length * num_kv_heads * head_dim
        * precision.bytes_per_param * batch_size
    )
    return kv_cache_bytes / (1024 ** 3)


def calc_activation_memory(
    hidden_size: int,
    num_layers: int,
    batch_size: int,
    seq_length: int,
    precision: Precision,
) -> float:
    """
    估算激活值占用的显存 (GB)
    简化估算：每层约 12 * hidden_size * seq_length * batch_size * bytes_per_param
    """
    act_bytes = (
        12 * hidden_size * seq_length * batch_size
        * precision.bytes_per_param * num_layers
    )
    return act_bytes / (1024 ** 3)


def calc_optimizer_memory(params_b: float, optimizer: str = "adam") -> float:
    """
    计算优化器状态占用的显存 (GB)
    Adam: 需要存储一阶矩和二阶矩，各 FP32（每参数 8 字节额外开销）
    SGD:  只需要动量，每参数 4 字节额外开销
    """
    if optimizer.lower() == "adam" or optimizer.lower() == "adamw":
        extra_bytes = params_b * 1e9 * 8  # m + v, 各 FP32
    elif optimizer.lower() == "sgd":
        extra_bytes = params_b * 1e9 * 4  # 动量
    else:
        extra_bytes = params_b * 1e9 * 8  # 默认按 Adam
    return extra_bytes / (1024 ** 3)


def calc_gradient_memory(params_b: float, precision: Precision) -> float:
    """计算梯度占用的显存 (GB)，与模型参数大小相同"""
    return calc_model_memory(params_b, precision)


def estimate_inference_memory(
    params_b: float,
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    precision: Precision,
    batch_size: int = 1,
    seq_length: int = 2048,
    num_kv_heads: Optional[int] = None,
) -> MemoryEstimate:
    """估算推理所需显存"""
    model_mem = calc_model_memory(params_b, precision)
    kv_mem = calc_kv_cache(
        hidden_size, num_layers, num_heads,
        batch_size, seq_length, precision, num_kv_heads
    )
    # 推理时激活值较小，只需要当前层
    act_mem = calc_activation_memory(
        hidden_size, 1, batch_size, seq_length, precision
    )
    # 额外开销（CUDA 上下文、碎片等），约 500MB - 1GB
    overhead = 0.5

    total = model_mem + kv_mem + act_mem + overhead

    return MemoryEstimate(
        model_params_gb=model_mem,
        kv_cache_gb=kv_mem,
        activation_gb=act_mem,
        optimizer_gb=0,
        gradient_gb=0,
        total_gb=total,
        description=f"推理模式 | 精度: {precision.label} | Batch: {batch_size} | 序列长度: {seq_length}",
    )


def estimate_training_memory(
    params_b: float,
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    precision: Precision,
    batch_size: int = 1,
    seq_length: int = 2048,
    optimizer: str = "adam",
    gradient_checkpointing: bool = False,
    num_kv_heads: Optional[int] = None,
) -> MemoryEstimate:
    """估算训练所需显存"""
    model_mem = calc_model_memory(params_b, precision)
    grad_mem = calc_gradient_memory(params_b, precision)
    opt_mem = calc_optimizer_memory(params_b, optimizer)

    act_layers = num_layers if not gradient_checkpointing else int(math.sqrt(num_layers))
    act_mem = calc_activation_memory(
        hidden_size, act_layers, batch_size, seq_length, precision
    )

    kv_mem = calc_kv_cache(
        hidden_size, num_layers, num_heads,
        batch_size, seq_length, precision, num_kv_heads
    )

    # 额外开销
    overhead = 1.0

    total = model_mem + grad_mem + opt_mem + act_mem + kv_mem + overhead

    desc = (
        f"训练模式 | 精度: {precision.label} | Batch: {batch_size} | "
        f"序列长度: {seq_length} | 优化器: {optimizer}"
    )
    if gradient_checkpointing:
        desc += " | 梯度检查点: 开启"

    return MemoryEstimate(
        model_params_gb=model_mem,
        kv_cache_gb=kv_mem,
        activation_gb=act_mem,
        optimizer_gb=opt_mem,
        gradient_gb=grad_mem,
        total_gb=total,
        description=desc,
    )


# ========== 常见 GPU 显存规格 ==========
GPU_SPECS = {
    "RTX 3060":    12,
    "RTX 3070":    8,
    "RTX 3080":    10,
    "RTX 3080Ti":  12,
    "RTX 3090":    24,
    "RTX 4060":    8,
    "RTX 4060Ti":  16,
    "RTX 4070":    12,
    "RTX 4070Ti":  12,
    "RTX 4080":    16,
    "RTX 4090":    24,
    "RTX 5090":    32,
    "A100-40GB":   40,
    "A100-80GB":   80,
    "A800-80GB":   80,
    "H100-80GB":   80,
    "H800-80GB":   80,
    "V100-16GB":   16,
    "V100-32GB":   32,
    "L40":         48,
    "A6000":       48,
}


def recommend_gpus(total_gb: float) -> list[dict]:
    """根据显存需求推荐 GPU 配置"""
    recommendations = []
    for gpu_name, vram in sorted(GPU_SPECS.items(), key=lambda x: x[1]):
        if vram >= total_gb:
            recommendations.append({"gpu": gpu_name, "count": 1, "total_vram": vram})
        elif total_gb <= vram * 2:
            recommendations.append({"gpu": gpu_name, "count": 2, "total_vram": vram * 2})
        elif total_gb <= vram * 4:
            recommendations.append({"gpu": gpu_name, "count": 4, "total_vram": vram * 4})
        elif total_gb <= vram * 8:
            recommendations.append({"gpu": gpu_name, "count": 8, "total_vram": vram * 8})
    return recommendations


def print_report(estimate: MemoryEstimate, model_name: str, show_gpu_rec: bool = True):
    """打印显存估算报告"""
    print("\n" + "=" * 60)
    print(f"  GPU 显存估算报告 - {model_name}")
    print("=" * 60)
    print(f"  {estimate.description}")
    print("-" * 60)
    print(f"  {'模型参数':　<16s} {format_bytes(estimate.model_params_gb):>12s}")
    if estimate.gradient_gb > 0:
        print(f"  {'梯度':　<16s} {format_bytes(estimate.gradient_gb):>12s}")
    if estimate.optimizer_gb > 0:
        print(f"  {'优化器状态':　<16s} {format_bytes(estimate.optimizer_gb):>12s}")
    print(f"  {'KV Cache':　<16s} {format_bytes(estimate.kv_cache_gb):>12s}")
    print(f"  {'激活值':　<16s} {format_bytes(estimate.activation_gb):>12s}")
    print(f"  {'额外开销':　<16s} {'~0.5-1 GB':>12s}")
    print("-" * 60)
    print(f"  {'预估总显存':　<16s} {format_bytes(estimate.total_gb):>12s}")
    print("=" * 60)

    if show_gpu_rec:
        recs = recommend_gpus(estimate.total_gb)
        if recs:
            print("\n  推荐 GPU 配置：")
            print("  " + "-" * 40)
            shown = 0
            for r in recs:
                if shown >= 5:
                    break
                if r["count"] == 1:
                    print(f"    {r['gpu']:>14s}  x {r['count']}  ({r['total_vram']} GB)")
                else:
                    print(f"    {r['gpu']:>14s}  x {r['count']}  ({r['total_vram']} GB 总计)")
                shown += 1
            print()
        else:
            print("\n  当前需求较大，建议使用多节点分布式训练。\n")


def list_models():
    """列出所有预设模型"""
    print("\n可用的预设模型：")
    print("-" * 50)
    for name, info in sorted(PRESET_MODELS.items()):
        print(f"  {name:<20s}  {info['params_b']:>6.1f}B 参数")
    print()


def list_gpus():
    """列出所有 GPU 规格"""
    print("\n已知 GPU 显存规格：")
    print("-" * 35)
    for name, vram in sorted(GPU_SPECS.items(), key=lambda x: x[1]):
        print(f"  {name:<16s}  {vram:>3d} GB")
    print()


def interactive_mode():
    """交互模式"""
    print("\n" + "=" * 50)
    print("  GPU 显存计算器 - 交互模式")
    print("=" * 50)

    # 选择模型
    print("\n选择模型：")
    print("  1. 从预设模型列表选择")
    print("  2. 手动输入模型参数")
    choice = input("\n请选择 (1/2): ").strip()

    if choice == "1":
        list_models()
        model_name = input("输入模型名称: ").strip().lower()
        if model_name not in PRESET_MODELS:
            print(f"错误：未找到模型 '{model_name}'")
            sys.exit(1)
        model = PRESET_MODELS[model_name]
        params_b = model["params_b"]
        hidden = model["hidden"]
        layers = model["layers"]
        heads = model["heads"]
    else:
        model_name = input("模型名称: ").strip() or "custom"
        params_b = float(input("参数量（十亿/B）: ").strip())
        hidden = int(input("隐藏层维度 (hidden_size): ").strip())
        layers = int(input("层数 (num_layers): ").strip())
        heads = int(input("注意力头数 (num_heads): ").strip())

    # 选择精度
    print("\n选择精度：")
    print("  1. FP32  (每参数 4 字节)")
    print("  2. FP16  (每参数 2 字节)")
    print("  3. BF16  (每参数 2 字节)")
    print("  4. INT8  (每参数 1 字节)")
    print("  5. INT4  (每参数 0.5 字节)")
    prec_map = {"1": Precision.FP32, "2": Precision.FP16, "3": Precision.BF16,
                "4": Precision.INT8, "5": Precision.INT4}
    prec_choice = input("请选择 (1-5, 默认 2): ").strip() or "2"
    precision = prec_map.get(prec_choice, Precision.FP16)

    # 选择模式
    print("\n选择使用场景：")
    print("  1. 推理")
    print("  2. 训练")
    mode = input("请选择 (1/2, 默认 1): ").strip() or "1"

    batch_size = int(input("Batch Size (默认 1): ").strip() or "1")
    seq_length = int(input("序列长度 (默认 2048): ").strip() or "2048")

    if mode == "2":
        optimizer = input("优化器 (adam/sgd, 默认 adam): ").strip() or "adam"
        gc = input("启用梯度检查点? (y/n, 默认 n): ").strip().lower()
        gradient_checkpointing = gc == "y"

        estimate = estimate_training_memory(
            params_b, hidden, layers, heads, precision,
            batch_size, seq_length, optimizer, gradient_checkpointing
        )
    else:
        estimate = estimate_inference_memory(
            params_b, hidden, layers, heads, precision,
            batch_size, seq_length
        )

    print_report(estimate, model_name)


def main():
    parser = argparse.ArgumentParser(
        description="GPU 显存计算器 - 根据模型参数估算所需显存",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 计算 LLaMA-7B 使用 FP16 推理所需显存
  python gpu_memory_calculator.py --model llama-7b --precision fp16

  # 计算 70B 模型使用 INT4 量化推理显存
  python gpu_memory_calculator.py --model llama2-70b --precision int4 --seq-length 4096

  # 计算自定义模型训练显存
  python gpu_memory_calculator.py --params 13 --hidden 5120 --layers 40 --heads 40 \\
      --precision bf16 --mode train --batch-size 4 --optimizer adam

  # 列出所有预设模型
  python gpu_memory_calculator.py --list-models

  # 交互模式
  python gpu_memory_calculator.py --interactive
        """,
    )

    parser.add_argument("--model", type=str, help="预设模型名称")
    parser.add_argument("--params", type=float, help="模型参数量（十亿/B）")
    parser.add_argument("--hidden", type=int, help="隐藏层维度")
    parser.add_argument("--layers", type=int, help="Transformer 层数")
    parser.add_argument("--heads", type=int, help="注意力头数")
    parser.add_argument("--kv-heads", type=int, default=None, help="KV 头数（GQA 模型使用）")
    parser.add_argument("--vocab", type=int, default=32000, help="词表大小（默认 32000）")

    parser.add_argument(
        "--precision", type=str, default="fp16",
        choices=["fp32", "fp16", "bf16", "int8", "int4"],
        help="模型精度（默认 fp16）",
    )
    parser.add_argument(
        "--mode", type=str, default="inference",
        choices=["inference", "train"],
        help="使用模式：inference(推理) 或 train(训练)",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch Size（默认 1）")
    parser.add_argument("--seq-length", type=int, default=2048, help="序列长度（默认 2048）")
    parser.add_argument(
        "--optimizer", type=str, default="adam",
        choices=["adam", "adamw", "sgd"],
        help="优化器类型（默认 adam）",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true", help="启用梯度检查点")
    parser.add_argument("--list-models", action="store_true", help="列出所有预设模型")
    parser.add_argument("--list-gpus", action="store_true", help="列出所有 GPU 规格")
    parser.add_argument("--interactive", action="store_true", help="交互模式")
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出")

    args = parser.parse_args()

    if args.list_models:
        list_models()
        return

    if args.list_gpus:
        list_gpus()
        return

    if args.interactive:
        interactive_mode()
        return

    # 确定模型参数
    if args.model:
        model_name = args.model.lower()
        if model_name not in PRESET_MODELS:
            print(f"错误：未找到模型 '{model_name}'，使用 --list-models 查看可用模型")
            sys.exit(1)
        model = PRESET_MODELS[model_name]
        params_b = model["params_b"]
        hidden = model["hidden"]
        layers = model["layers"]
        heads = model["heads"]
    elif args.params and args.hidden and args.layers and args.heads:
        model_name = f"custom-{args.params}B"
        params_b = args.params
        hidden = args.hidden
        layers = args.layers
        heads = args.heads
    else:
        parser.print_help()
        print("\n错误：请指定 --model 或提供完整的模型参数（--params, --hidden, --layers, --heads）")
        sys.exit(1)

    # 精度映射
    precision_map = {
        "fp32": Precision.FP32,
        "fp16": Precision.FP16,
        "bf16": Precision.BF16,
        "int8": Precision.INT8,
        "int4": Precision.INT4,
    }
    precision = precision_map[args.precision]

    # 计算显存
    if args.mode == "train":
        estimate = estimate_training_memory(
            params_b, hidden, layers, heads, precision,
            args.batch_size, args.seq_length, args.optimizer,
            args.gradient_checkpointing, args.kv_heads,
        )
    else:
        estimate = estimate_inference_memory(
            params_b, hidden, layers, heads, precision,
            args.batch_size, args.seq_length, args.kv_heads,
        )

    if args.json:
        result = {
            "model": model_name,
            "mode": args.mode,
            "precision": precision.label,
            "batch_size": args.batch_size,
            "seq_length": args.seq_length,
            "memory": {
                "model_params_gb": round(estimate.model_params_gb, 2),
                "kv_cache_gb": round(estimate.kv_cache_gb, 2),
                "activation_gb": round(estimate.activation_gb, 2),
                "optimizer_gb": round(estimate.optimizer_gb, 2),
                "gradient_gb": round(estimate.gradient_gb, 2),
                "total_gb": round(estimate.total_gb, 2),
            },
            "gpu_recommendations": recommend_gpus(estimate.total_gb)[:5],
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print_report(estimate, model_name)


if __name__ == "__main__":
    main()
