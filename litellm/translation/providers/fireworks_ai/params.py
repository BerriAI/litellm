"""Parameter gates for the fireworks_ai serializer.

v1's gate is ``_check_valid_arg`` over ``FireworksAIConfig.
get_supported_openai_params`` — its OWN static list (stream, max_tokens,
mct, temperature, top_p, top_k, penalties, n, stop, response_format, user,
logprobs, prompt_truncate_length, context_length_exceeded_behavior) plus
THREE capability forks over the ``fireworks_ai/{model}`` map keys:
``tools``+``parallel_tool_calls`` iff supports_function_calling,
``tool_choice`` iff supports_tool_choice, ``reasoning_effort`` iff
supports_reasoning. ``user`` is SERVED verbatim (explicitly listed —
contrast the base-list providers). The list's ``top_k`` mention is DEAD at
runtime: top_k is not an OpenAI param so it never reaches
``_check_valid_arg``/the map — fireworks_ai IS in
``openai_compatible_providers``, so it rides the extra_body crossing
(wire-proven in the request gate), the shared default fallback arm.

CAPABILITY DRIFT (probed at HEAD, the gate's one-direction soundness):
v1's ``supports_function_calling`` answer for fireworks is NOT the map-row
flag — ``FireworksAIConfig.get_provider_info`` defaults it TRUE (even for
unknown models) and ``_get_model_cost_capability`` overrides it via a
hyphen-boundary longest-match SCAN over the whole fireworks map index. The
deps read (the map flag) is strictly NARROWER: 215/246 chat rows answer
v1-True/v2-False and ZERO rows answer v1-False/v2-True (probed; the mirror
gate pins the direction). v2 therefore serves tools only on the
explicitly-flagged rows and falls back elsewhere with a note naming this
drift — v1 serves OR raises per its own gate, so the fallback reproduces
v1 either way. Porting the scan would need a deps surface over the whole
model map (re-evaluate if fireworks tools fallback volume hurts).

Two v1 rewrites fall back typed so v1 serves its own machinery:

- ``response_format`` WITH tools: v1's ``_add_response_format_to_tools``
  (is_response_format_supported=False) pops response_format and sets the
  ``json_mode`` routing marker — the groq-precedent cross-plane feature
  (request map + response tool->content conversion).
- tool ``parameters`` carrying legacy def blocks (top-level ``definitions``
  or ``components.schemas`` — v1's ``_has_legacy_defs``): v1 inlines the
  ``$ref``s via ``unpack_legacy_defs`` (byte-capped schema surgery). That
  fallback fires at the SHARED inbound boundary (parse.py's "legacy $defs
  ... need v1's schema inlining" semantic check) — no provider arm needed,
  pinned in the request gate.
"""

from __future__ import annotations

from ...deps import TranslationDeps
from ...ir import ChatRequest
from ..compat_sdk.checks import unsupported_against
from ..openai_compat.params import unsupported_response_format

_FIREWORKS_BASE = frozenset(
    {
        "stream",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "stop",
        "response_format",
        "user",
    }
)
_FC_KEYS = frozenset({"tools", "parallel_tool_calls"})


def supports_fireworks_tools(model: str, deps: TranslationDeps) -> bool:
    return deps.supports_capability(
        f"fireworks_ai/{model}", "supports_function_calling"
    )


def supports_fireworks_tool_choice(model: str, deps: TranslationDeps) -> bool:
    return deps.supports_capability(f"fireworks_ai/{model}", "supports_tool_choice")


def supports_fireworks_reasoning(model: str, deps: TranslationDeps) -> bool:
    return deps.supports_capability(f"fireworks_ai/{model}", "supports_reasoning")


def fireworks_allowed(model: str, deps: TranslationDeps) -> frozenset[str]:
    allowed = _FIREWORKS_BASE
    if supports_fireworks_tools(model, deps):
        allowed = allowed | _FC_KEYS
    if supports_fireworks_tool_choice(model, deps):
        allowed = allowed | {"tool_choice"}
    if supports_fireworks_reasoning(model, deps):
        allowed = allowed | {"reasoning_effort"}
    return allowed


def unsupported_params(request: ChatRequest, deps: TranslationDeps) -> str | None:
    if request.response_format.is_some() and len(request.tools) > 0:
        return (
            "response_format together with tools on fireworks_ai: v1's "
            "_add_response_format_to_tools pops response_format and sets the "
            "json_mode routing marker (request+response cross-plane "
            "machinery); v1 serves it"
        )
    capability_note = (
        "on fireworks_ai: v2's capability read (the fireworks_ai/{m} map "
        "flag) is strictly narrower than v1's get_provider_info default-true "
        "+ hyphen-boundary scan — v1 serves or raises per its own gate, and "
        "the typed fallback reproduces it either way"
    )
    return unsupported_against(
        request,
        provider="fireworks_ai",
        allowed=fireworks_allowed(request.model, deps),
        notes={
            "tools": f"tools {capability_note}",
            "parallel_tool_calls": f"parallel_tool_calls {capability_note}",
            "tool_choice": f"tool_choice {capability_note}",
            "reasoning_effort": f"reasoning_effort {capability_note}",
        },
    ) or unsupported_response_format(request)
