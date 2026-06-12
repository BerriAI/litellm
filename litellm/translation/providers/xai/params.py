"""Parameter gates for the xai (Grok) serializer.

v1's gate is ``_check_valid_arg`` over ``XAIChatConfig.
get_supported_openai_params`` (utils.py:4162-4203): an unsupported param
RAISES ``UnsupportedParamsError`` unless ``drop_params``, in which case it is
popped BEFORE ``map_openai_params`` runs — so the ``max_completion_tokens ->
max_tokens`` rename arm (x_t:185-186) is dead code on the standard path.
v2 mirrors the SUPPORTED-LIST truth, never the raise-vs-drop interplay:
every shape v1 raises or drops on falls back typed so v1 serves its own
error (the v2-openai response_format precedent).

Per-model gates mirror x_t:155-174 (substring checks) and the
``litellm.supports_reasoning(model, "xai")`` model-map read, reproduced
through ``deps.supports_capability`` over the ``xai/{model}`` map key (the
bare wire model has no model-map row; verified in-process at HEAD).
"""

from __future__ import annotations

from ...deps import TranslationDeps
from ...ir import ChatRequest

_NO_STOP_FAMILIES = ("grok-3-mini", "grok-4", "grok-code-fast")


def supports_stop(model: str) -> bool:
    return not any(family in model for family in _NO_STOP_FAMILIES)


def supports_reasoning(model: str, deps: TranslationDeps) -> bool:
    return deps.supports_capability(f"xai/{model}", "supports_reasoning")


def unsupported_params(request: ChatRequest, deps: TranslationDeps) -> str | None:
    if request.params.max_completion_tokens.is_some():
        return (
            "max_completion_tokens is outside xai's supported list; v1's "
            "get_optional_params raises UnsupportedParamsError (or drops it "
            "under drop_params) before the rename arm can run"
        )
    if request.params.top_k.is_some():
        return "top_k is not an xai chat param; v1's get_optional_params raises or drops it"
    if request.thinking.is_some():
        return "thinking is not an xai chat param; v1's get_optional_params raises or drops it"
    if len(request.params.stop) > 0 and not supports_stop(request.model):
        return (
            f"stop on {request.model}: outside xai's supported list "
            "(grok-3-mini/grok-4/grok-code-fast); v1 raises or drops it"
        )
    if request.reasoning_effort.is_some() and not supports_reasoning(
        request.model, deps
    ):
        return (
            f"reasoning_effort on non-reasoning xai model {request.model}; "
            "v1 raises or drops it"
        )
    return None
