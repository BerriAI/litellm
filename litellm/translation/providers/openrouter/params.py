"""Parameter gates for the openrouter serializer.

v1's gate is ``_check_valid_arg`` over ``OpenrouterConfig.
get_supported_openai_params`` — the OpenAI base list plus ``thinking`` and
``reasoning_effort`` for reasoning-capable models (DUAL-keyed lookup:
``supports_reasoning(model, "openrouter") OR supports_reasoning(model)``,
error-swallowed; openrouter/chat/transformation.py:36-49). openrouter is NOT
in ``litellm.openai_compatible_providers``, so ``get_optional_params`` packs
non-OpenAI kwargs TOP-LEVEL (the else arm of
``add_provider_specific_params_to_optional_params``) — ``top_k`` rides
verbatim onto the wire (NOT extra_body; a dossier-drift fact, wire-proven in
the request gate), so it is SERVED here.

``thinking`` falls back BOTH ways: on capable models v1 serves the wire dict
VERBATIM (an unported emission — the zai/minimax precedent: the IR's
ThinkingParam cannot round-trip explicit-null/extra-key shapes
byte-identically) and on non-capable models v1 raises UnsupportedParamsError.
``reasoning_effort`` is SERVED on capable models (verbatim key emission, the
xai shape) and falls back where v1 raises.
"""

from __future__ import annotations

from ...deps import TranslationDeps
from ...ir import ChatRequest
from ..compat_sdk.checks import BASE_LIST, unsupported_against, user_note
from ..openai_compat.params import unsupported_response_format

_OPENROUTER_LIST = BASE_LIST | frozenset({"top_k", "reasoning_effort"})

CACHE_CONTROL_MODEL_SUBSTRINGS = ("claude", "gemini", "minimax", "glm", "z-ai")
"""v1's CacheControlSupportedModels enum (lower-cased substring match):
these models KEEP cache_control on the wire and get message-level
cache_control MOVED into the last content block — the guard falls back on
them; every other model gets the base recursive strip, which the IR's
cache_control drop reproduces byte-identically."""


def supports_cache_control_in_content(model: str) -> bool:
    lowered = model.lower()
    return any(part in lowered for part in CACHE_CONTROL_MODEL_SUBSTRINGS)


def supports_openrouter_reasoning(model: str, deps: TranslationDeps) -> bool:
    """v1's dual-keyed lookup. Models whose id ALREADY starts with
    ``openrouter/`` answer False in v1 (get_model_info strips the duplicate
    provider prefix, so the openrouter/openrouter/{auto,free} map rows are
    unreachable — probed at HEAD); mirror that instead of reading the map
    rows v1 cannot reach."""
    if model.startswith("openrouter/"):
        return False
    return deps.supports_capability(
        f"openrouter/{model}", "supports_reasoning"
    ) or deps.supports_capability(model, "supports_reasoning")


def unsupported_params(request: ChatRequest, deps: TranslationDeps) -> str | None:
    reasoning_capable = supports_openrouter_reasoning(request.model, deps)
    if request.thinking.is_some():
        if reasoning_capable:
            return (
                "thinking on a reasoning-capable openrouter model: v1 serves "
                "the wire dict VERBATIM (an unported emission; the IR cannot "
                "round-trip every accepted shape byte-identically)"
            )
        return (
            "thinking on openrouter: outside v1's supported list for "
            "non-reasoning models; get_optional_params raises "
            "UnsupportedParamsError (or drops it under drop_params)"
        )
    if request.reasoning_effort.is_some() and not reasoning_capable:
        return (
            "reasoning_effort on openrouter: outside v1's supported list for "
            "non-reasoning models; get_optional_params raises "
            "UnsupportedParamsError (or drops it under drop_params)"
        )
    return unsupported_against(
        request,
        provider="openrouter",
        allowed=_OPENROUTER_LIST,
        notes={"user": user_note("openrouter")},
    ) or unsupported_response_format(request)
