"""Parameter gates for the hosted_vllm serializer.

v1's gate is ``_check_valid_arg`` over ``HostedVLLMChatConfig.
get_supported_openai_params`` — the OpenAI base list plus ``thinking`` and
``reasoning_effort``, appended UNCONDITIONALLY for every model (no
capability fork; hosted_vllm/chat/transformation.py:90-93). Both are
SERVED: ``reasoning_effort`` verbatim, ``thinking`` through v1's
deterministic budget-band rewrite (serialize.py). The provider-shaped v1
rewrites that fall back here do so at the SHARED boundaries, no provider
arm needed (pinned in the request gate): custom-type tools (openai guard;
v1 synthesizes function tools), assistant ``thinking_blocks`` (openai
guard; v1 prepends them as content blocks), and ``file`` content parts
(inbound boundary; v1 converts video files to ``video_url`` blocks).
"""

from __future__ import annotations

from ...deps import TranslationDeps
from ...ir import ChatRequest
from ..compat_sdk.checks import BASE_LIST, unsupported_against, user_note
from ..openai_compat.params import unsupported_response_format

_HOSTED_VLLM_LIST = BASE_LIST | frozenset({"thinking", "reasoning_effort"})


def unsupported_params(request: ChatRequest, deps: TranslationDeps) -> str | None:
    return unsupported_against(
        request,
        provider="hosted_vllm",
        allowed=_HOSTED_VLLM_LIST,
        notes={"user": user_note("hosted_vllm")},
    ) or unsupported_response_format(request)
