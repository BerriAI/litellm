"""hosted_vllm chat-completion response JSON -> IR ``ChatResponse``.

``HostedVLLMChatConfig`` has no ``transform_response`` override: the
inherited base GPT transform is LIVE on the dedicated elif over a FRESH
``ModelResponse`` (model=None), so the response model is the BARE wire
model (the xai R4 / deepseek pattern; no ``hosted_vllm/`` prefix — pinned
by test_differential_hosted_vllm_response.py). The shared openai parser is
that normalizer's mirror, verbatim.
"""

from __future__ import annotations

from ..openai_compat.response import parse_response

__all__ = ("parse_response",)
