"""Per-provider parameter gates for the httpx-path openai-compat shims.

These nine providers have dedicated ``completion()`` elifs (heroku 2229,
bedrock_mantle 2364, minimax 2587, amazon_nova 3176, compactifai 3249,
datarobot 3355, gradient_ai 4421, lemonade 4469, ovhcloud 4498 at HEAD) into
``base_llm_http_handler`` — the xai routing shape, NOT the SDK preset/
re-prefix shape. The v1 param gate is the same RAISE-unless-drop_params
``_check_valid_arg`` chain the compat_sdk family mirrors, so the generic
checker is shared from there.

lemonade is the one dual-path member: its chat config is UNREGISTERED in
``ProviderConfigManager._PROVIDER_CONFIG_MAP`` at HEAD, so param resolution
falls back to the openai supported list + the bare ``OpenAILikeChatConfig``
map arm (mct RENAMES), while the transform side is live via the explicit
``provider_config=lemonade_transformation`` in its elif. The IR-visible
surface is registration-invariant (base list + mct rename both ways) —
pinned by the lemonade facts canary in the request differential.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Literal

from ...deps import TranslationDeps
from ...ir import ChatRequest
from ..compat_sdk.checks import (
    BASE_LIST,
    base_list_unsupported,
    unsupported_against,
)
from ..openai_compat.params import unsupported_response_format

_GateFn = Callable[[ChatRequest, TranslationDeps], "str | None"]

CompatHttpxProvider = Literal[
    "heroku",
    "bedrock_mantle",
    "minimax",
    "compactifai",
    "amazon_nova",
    "datarobot",
    "gradient_ai",
    "ovhcloud",
    "lemonade",
]

# BedrockMantleChatConfig: OpenAILike base list + reasoning_effort iff
# supports_reasoning("bedrock_mantle/{m}") (the cerebras pattern; verified
# in-process: served on openai.gpt-oss-120b, UnsupportedParamsError on an
# unflagged model).
_BEDROCK_MANTLE_LIST = BASE_LIST | frozenset({"reasoning_effort"})


def supports_bedrock_mantle_reasoning(model: str, deps: TranslationDeps) -> bool:
    return deps.supports_capability(f"bedrock_mantle/{model}", "supports_reasoning")


def bedrock_mantle_unsupported(
    request: ChatRequest, deps: TranslationDeps
) -> str | None:
    allowed = (
        _BEDROCK_MANTLE_LIST
        if supports_bedrock_mantle_reasoning(request.model, deps)
        else _BEDROCK_MANTLE_LIST - {"reasoning_effort"}
    )
    return (
        unsupported_against(
            request,
            provider="bedrock_mantle",
            allowed=allowed,
            notes={
                "reasoning_effort": (
                    "reasoning_effort on non-reasoning bedrock_mantle model "
                    f"{request.model} (model-map supports_reasoning gate); "
                    "v1 raises or drops it"
                )
            },
        )
        # the OpenAILike base list carries the gpt-4/gpt-3.5-turbo-16k
        # response_format name gate (super() chain)
        or unsupported_response_format(request)
    )


# AmazonNovaChatConfig's STATIC list (no capability gates): response_format
# and parallel_tool_calls RAISE; reasoning_effort is unconditional (emission
# derived from membership here); ``metadata`` is in v1's list but outside the
# IR (inbound fallback, v1 serves it).
_AMAZON_NOVA_LIST = frozenset(
    {
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "stop",
        "stream",
        "tools",
        "tool_choice",
        "reasoning_effort",
    }
)


def amazon_nova_unsupported(request: ChatRequest, deps: TranslationDeps) -> str | None:
    return unsupported_against(
        request, provider="amazon_nova", allowed=_AMAZON_NOVA_LIST
    )


# GradientAIConfig: own list (frequency/presence penalties and the kb_*
# retrieval params are outside the IR); its map does NOT call super and never
# renames mct (verbatim); the map carries its OWN UnsupportedParamsError
# raise that fires even for params ``_check_valid_arg`` skips — ``user``
# RAISES on gradient_ai (verified in-process), unlike the family's silent
# drop.
_GRADIENT_AI_LIST = frozenset(
    {"max_tokens", "max_completion_tokens", "temperature", "top_p", "stream", "stop"}
)


def gradient_ai_unsupported(request: ChatRequest, deps: TranslationDeps) -> str | None:
    return unsupported_against(
        request,
        provider="gradient_ai",
        allowed=_GRADIENT_AI_LIST,
        notes={
            "user": (
                "user on gradient_ai: GradientAIConfig.map_openai_params "
                "raises its own UnsupportedParamsError on any param outside "
                "its list, including user (which _check_valid_arg skips); "
                "v1 raises"
            )
        },
    )


def minimax_unsupported(request: ChatRequest, deps: TranslationDeps) -> str | None:
    """Base list; ``thinking`` is in v1's list for supports_reasoning
    ("minimax/{m}") models and copied verbatim top-level — unported emission,
    typed fallback (v1 serves it). ``reasoning_split`` is provider-native,
    outside the IR."""
    return unsupported_against(
        request,
        provider="minimax",
        allowed=BASE_LIST,
        notes={
            "thinking": (
                "thinking on minimax: v1 copies the verbatim dict top-level "
                "for supports_reasoning models (capability-gated) and raises "
                "otherwise; that emission is unported, v1 handles it"
            )
        },
    ) or unsupported_response_format(request)


def _base_gate(provider: str) -> _GateFn:
    def gate(request: ChatRequest, deps: TranslationDeps) -> str | None:
        return base_list_unsupported(request, deps, provider)

    return gate


heroku_unsupported = _base_gate("heroku")
compactifai_unsupported = _base_gate("compactifai")
datarobot_unsupported = _base_gate("datarobot")
ovhcloud_unsupported = _base_gate("ovhcloud")
lemonade_unsupported = _base_gate("lemonade")

ALLOWED: Mapping[CompatHttpxProvider, frozenset[str]] = MappingProxyType(
    {
        "heroku": BASE_LIST,
        "bedrock_mantle": _BEDROCK_MANTLE_LIST,
        "minimax": BASE_LIST,
        "compactifai": BASE_LIST,
        "amazon_nova": _AMAZON_NOVA_LIST,
        "datarobot": BASE_LIST,
        "gradient_ai": _GRADIENT_AI_LIST,
        "ovhcloud": BASE_LIST,
        "lemonade": BASE_LIST,
    }
)
