"""
国内模型兼容工具包

提供国内模型（阿里云、火山引擎、MiniMax、小米、DeepSeek、Moonshot、智谱）
的兼容判断函数，用于在 Responses API → Chat Completions 转换时过滤不支持的字段。
"""

from litellm.llms.domestic.domestic_utils import (
    is_domestic_model,
    is_domestic_endpoint,
    is_domestic_model_or_endpoint,
)

__all__ = [
    "is_domestic_model",
    "is_domestic_endpoint",
    "is_domestic_model_or_endpoint",
]
