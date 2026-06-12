"""Model-family and parameter gates for the OpenAI passthrough serializer.

v1 dispatches o-series and gpt-5 models onto separate config classes
(``OpenAIOSeriesConfig`` / ``OpenAIGPT5Config``) with their own param
rewrites: max_tokens -> max_completion_tokens, temperature laws,
reasoning_effort normalization, system -> user rewrites. Those families fail
closed here until they are ported; the name checks mirror v1's own
(``is_model_o_series_model`` without the model-map gate, which only ever
NARROWS v1's match, so v2 can only over-fallback, never diverge).
"""

from __future__ import annotations

from ...ir import ChatRequest

_RESPONSE_FORMAT_UNSUPPORTED_MODELS = ("gpt-4", "gpt-3.5-turbo-16k")
"""v1's get_supported_openai_params excludes response_format for exactly
these two model names (gpt_transformation.py:172-175) and then DROPS or
RAISES on it in get_optional_params; fail closed instead of re-deriving the
drop_params interplay."""


def unsupported_model_family(model: str) -> str | None:
    base = model.split("/")[-1]
    if len(base) > 1 and base[0] == "o" and base[1].isdigit():
        return (
            f"o-series model {model}: v1's OpenAIOSeriesConfig owns its param rewrites"
        )
    if "gpt-5" in model and not base.startswith("gpt-5-chat"):
        return f"gpt-5 model {model}: v1's OpenAIGPT5Config owns its param rewrites"
    return None


def unsupported_params(request: ChatRequest) -> str | None:
    if request.params.top_k.is_some():
        return "top_k is not an OpenAI chat param; v1's get_optional_params raises or drops it"
    if request.thinking.is_some():
        return "thinking on a plain-GPT OpenAI model; v1's get_optional_params raises or drops it"
    if request.reasoning_effort.is_some():
        return (
            "reasoning_effort on a plain-GPT OpenAI model; "
            "v1's get_optional_params raises or drops it"
        )
    if request.user.is_some():
        return (
            "user param is gated on litellm.open_ai_chat_completion_models "
            "membership in v1 (gpt_transformation.py:177-187); v1 handles it"
        )
    return None


def unsupported_response_format(request: ChatRequest) -> str | None:
    if request.response_format.is_none():
        return None
    if request.model in _RESPONSE_FORMAT_UNSUPPORTED_MODELS:
        return (
            f"response_format on {request.model}: outside v1's supported set "
            "(gpt_transformation.py:172-175); v1 raises or drops it"
        )
    return None
