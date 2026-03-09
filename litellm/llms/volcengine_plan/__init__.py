"""
Volcengine Plan LLM Provider
Support for Volcengine Plan (ByteDance) chat models via /api/coding/v3 endpoint.
Shares API key with Volcengine provider.
"""

from .chat.transformation import VolcEnginePlanChatConfig

__all__ = [
    "VolcEnginePlanChatConfig",
]
