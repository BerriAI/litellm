import json
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest

REPO_ROOT: Final = Path(__file__).parents[2]
PRICE_FILES: Final = (
    REPO_ROOT / "model_prices_and_context_window.json",
    REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json",
)
SOURCE: Final = "https://wandb.ai/site/pricing/tokens/"
CURRENT_PRICES_PER_MILLION_TOKENS: Final = MappingProxyType(
    {
        "wandb/deepseek-ai/DeepSeek-R1-0528": (1.35, 5.40, None),
        "wandb/deepseek-ai/DeepSeek-V3-0324": (1.14, 2.75, None),
        "wandb/deepseek-ai/DeepSeek-V3.1": (0.55, 1.65, None),
        "wandb/deepseek-ai/DeepSeek-V4-Flash": (0.14, 0.28, 0.07),
        "wandb/deepseek-ai/DeepSeek-V4-Flash-0731": (0.13, 0.28, 0.07),
        "wandb/deepseek-ai/DeepSeek-V4-Pro": (1.15, 2.55, 0.20),
        "wandb/google/gemma-4-31B-it": (0.10, 0.34, None),
        "wandb/ibm-granite/granite-4.1-8b": (0.05, 0.10, None),
        "wandb/JetBrains/Mellum2-12B-A2.5B-Instruct": (0.05, 0.10, None),
        "wandb/meta-llama/Llama-3.1-70B-Instruct": (0.80, 0.80, None),
        "wandb/meta-llama/Llama-3.1-8B-Instruct": (0.22, 0.22, None),
        "wandb/meta-llama/Llama-3.3-70B-Instruct": (0.71, 0.71, None),
        "wandb/meta-llama/Llama-4-Scout-17B-16E-Instruct": (0.17, 0.66, None),
        "wandb/microsoft/Phi-4-mini-instruct": (0.08, 0.35, None),
        "wandb/MiniMaxAI/MiniMax-M2.5": (0.30, 1.20, None),
        "wandb/MiniMaxAI/MiniMax-M3": (0.23, 0.96, 0.05),
        "wandb/moonshotai/Kimi-K2.6": (0.65, 3.41, 0.15),
        "wandb/moonshotai/Kimi-K2.7-Code": (0.71, 3.50, 0.15),
        "wandb/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8": (0.20, 0.80, None),
        "wandb/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B": (0.75, 2.75, 0.15),
        "wandb/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B": (0.10, 0.25, 0.05),
        "wandb/openai/gpt-oss-120b": (0.03, 0.17, None),
        "wandb/openai/gpt-oss-20b": (0.03, 0.13, None),
        "wandb/OpenPipe/Qwen3-14B-Instruct": (0.05, 0.22, None),
        "wandb/Qwen/Qwen3-235B-A22B-Instruct-2507": (0.10, 0.10, None),
        "wandb/Qwen/Qwen3-235B-A22B-Thinking-2507": (0.10, 0.10, None),
        "wandb/Qwen/Qwen3-30B-A3B-Instruct-2507": (0.10, 0.30, None),
        "wandb/Qwen/Qwen3-Coder-480B-A35B-Instruct": (1.00, 1.50, None),
        "wandb/Qwen/Qwen3.5-35B-A3B": (0.25, 1.25, None),
        "wandb/Qwen/Qwen3.6-27B": (0.60, 3.60, 0.12),
        "wandb/Qwen/Qwen3.6-35B-A3B": (0.25, 1.25, None),
        "wandb/zai-org/GLM-4.5": (0.55, 2.00, None),
        "wandb/zai-org/GLM-5.1": (1.40, 4.40, 0.26),
        "wandb/zai-org/GLM-5.2": (0.76, 2.42, 0.14),
    }
)
CONFIRMED_CAPABILITIES: Final = MappingProxyType(
    {
        "wandb/google/gemma-4-31B-it": ("supports_reasoning", "supports_vision"),
        "wandb/MiniMaxAI/MiniMax-M3": ("supports_vision",),
        "wandb/moonshotai/Kimi-K2.6": ("supports_reasoning", "supports_vision"),
        "wandb/moonshotai/Kimi-K2.7-Code": ("supports_vision",),
    }
)


@pytest.mark.parametrize("price_file", PRICE_FILES)
def test_current_wandb_pricing(price_file: Path) -> None:
    prices: Final = json.loads(price_file.read_text())

    for model, (input_price, output_price, cache_price) in CURRENT_PRICES_PER_MILLION_TOKENS.items():
        metadata: Final = prices[model]
        assert metadata["input_cost_per_token"] == pytest.approx(input_price / 1_000_000)
        assert metadata["output_cost_per_token"] == pytest.approx(output_price / 1_000_000)
        assert metadata.get("cache_read_input_token_cost") == (
            pytest.approx(cache_price / 1_000_000) if cache_price is not None else None
        )
        assert metadata["source"] == SOURCE


@pytest.mark.parametrize("price_file", PRICE_FILES)
def test_confirmed_wandb_model_capabilities(price_file: Path) -> None:
    prices: Final = json.loads(price_file.read_text())

    for model, capabilities in CONFIRMED_CAPABILITIES.items():
        for capability in capabilities:
            assert prices[model][capability] is True
