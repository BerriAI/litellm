"""
国内模型兼容判断工具

独立模块，避免循环导入问题。
用于判断请求是否来自国内模型 provider，需要兼容处理。

退出选项：
- 设置环境变量 LITELLM_DISABLE_DOMESTIC_COMPATIBILITY=true 可完全禁用兼容过滤
- 这允许用户即使模型名包含 "deepseek" 等关键词，也能使用完整的 OpenAI 参数
"""

import os
from typing import Optional


def _is_domestic_compatibility_disabled() -> bool:
    """
    Check if domestic compatibility filtering is explicitly disabled via environment variable.
    This provides an opt-out mechanism for users who want full OpenAI compatibility
    even for models that match domestic patterns.

    Returns:
        bool: True if domestic compatibility is disabled
    """
    return os.environ.get("LITELLM_DISABLE_DOMESTIC_COMPATIBILITY", "").lower() in (
        "true",
        "1",
        "yes",
    )


def is_domestic_model(model_name: Optional[str]) -> bool:
    """
    Check if the model is a domestic (Chinese) model based on its name.
    Domestic models don't support OpenAI's native Responses API format
    and certain Codex CLI specific parameters/tools.

    Args:
        model_name: Model name (e.g., "qwen3.5-plus", "MiniMax-M2.7", "openai/MiniMax-M2.7")

    Returns:
        bool: True if it's a domestic model that needs compatibility handling
    """
    # 退出选项：如果用户明确禁用兼容过滤，返回 False
    if _is_domestic_compatibility_disabled():
        return False

    # 类型检查：只接受字符串类型，排除 mock 函数对象等
    if not model_name or not isinstance(model_name, str):
        return False

    model_lower = model_name.lower()

    # Domestic model name patterns (covers both raw names and prefixed names like "openai/MiniMax-M2.7")
    domestic_patterns = [
        "qwen",  # Alibaba Qwen series
        "glm",  # Zhipu GLM series
        "doubao",  # Volcengine (ByteDance) Doubao series
        "minimax",  # MiniMax series
        "mimo",  # Xiaomi MiMo series
        "deepseek",  # DeepSeek series
        "kimi",  # Moonshot Kimi series
    ]

    return any(pattern in model_lower for pattern in domestic_patterns)


def is_domestic_endpoint(api_base: Optional[str]) -> bool:
    """
    Check if the api_base is a domestic (Chinese) model provider endpoint.
    Used as fallback when model name doesn't contain domestic pattern.

    Args:
        api_base: API endpoint URL

    Returns:
        bool: True if it's a domestic endpoint
    """
    # 退出选项：如果用户明确禁用兼容过滤，返回 False
    if _is_domestic_compatibility_disabled():
        return False

    # 类型检查：只接受字符串类型，排除 mock 函数对象等
    if not api_base or not isinstance(api_base, str):
        return False

    domestic_endpoints = [
        "dashscope.aliyuncs.com",  # Alibaba DashScope
        "ark.cn-beijing.volces.com",  # Volcengine (ByteDance)
        "api.minimaxi.com",  # MiniMax official
        "xiaomimimo.com",  # Xiaomi MiMo
        "api.deepseek.com",  # DeepSeek official
        "moonshot.cn",  # Moonshot Kimi official
        "bigmodel.cn",  # Zhipu GLM official
    ]

    return any(endpoint in str(api_base) for endpoint in domestic_endpoints)


def is_domestic_model_or_endpoint(
    model_name: Optional[str], api_base: Optional[str] = None
) -> bool:
    """
    Check if either model name or endpoint indicates a domestic model.
    This covers both cases:
    1. Model name contains domestic pattern (e.g., "MiniMax-M2.7", "openai/qwen3.5-plus")
    2. Endpoint is domestic provider (when model name is just group name like "codex-model")

    Args:
        model_name: Model name
        api_base: API endpoint URL (optional)

    Returns:
        bool: True if it's a domestic model/endpoint
    """
    if is_domestic_model(model_name):
        return True
    if api_base and is_domestic_endpoint(api_base):
        return True
    return False
