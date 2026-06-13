"""Parameter gates for the groq chat serializer.

v1's gate is ``_check_valid_arg`` over ``GroqChatConfig.
get_supported_openai_params`` (the base list minus ``max_retries``, plus
``reasoning_effort`` iff ``litellm.supports_reasoning(model, "groq")``) and
the json_schema THREE-WAY fork in ``map_openai_params`` (probed in-process
at HEAD):

- a model with the ``groq/{m}`` ``supports_response_schema`` map flag
  (llama-4, gpt-oss, kimi-k2-0905 rows) passes ``response_format`` with a
  ``json_schema`` VERBATIM — served;
- a NON-native model + json_schema + user ``tools`` raises
  ``litellm.BadRequestError`` (NOT UnsupportedParamsError — the wave's one
  different request raise class, pinned in the gate);
- a non-native model + json_schema alone gets the json_tool_call
  WORKAROUND (tools + forced tool_choice + json_mode + fake_stream — a
  cross-plane rewrite spanning request, response, and routing): typed
  fallback, v1 serves it end-to-end (researcher-4's recommended shape).

``response_format: json_object`` is served verbatim (the fake_stream it
sets is a routing key hh pops before the wire). ``thinking`` raises;
``user`` is silently dropped; n/seed/penalties/logit_bias/logprobs/
top_logprobs/service_tier/web_search_options pass through in v1 but are
parse-level unknowns in v2 (typed fallback, v1 serves).
"""

from __future__ import annotations

from ...deps import TranslationDeps
from ...ir import ChatRequest


def supports_reasoning(model: str, deps: TranslationDeps) -> bool:
    return deps.supports_capability(f"groq/{model}", "supports_reasoning")


def _supports_response_schema(model: str, deps: TranslationDeps) -> bool:
    return deps.supports_capability(f"groq/{model}", "supports_response_schema")


def unsupported_params(request: ChatRequest, deps: TranslationDeps) -> str | None:
    if request.thinking.is_some():
        return "thinking is not a groq chat param; v1 raises or drops it"
    if request.user.is_some():
        return (
            "user is model-list gated upstream and silently dropped for "
            "groq; v1 serves its own drop"
        )
    if request.reasoning_effort.is_some() and not supports_reasoning(
        request.model, deps
    ):
        return (
            f"reasoning_effort on non-reasoning groq model {request.model}; "
            "v1 raises or drops it"
        )
    return _response_format_reason(request, deps)


def _response_format_reason(request: ChatRequest, deps: TranslationDeps) -> str | None:
    response_format = request.response_format.default_value(None)
    if response_format is None or response_format.tag != "json_schema":
        return None
    if _supports_response_schema(request.model, deps):
        return None  # native passthrough — served verbatim
    if len(request.tools) > 0:
        return (
            f"response_format json_schema with tools on non-native groq "
            f"model {request.model}: v1 raises litellm.BadRequestError "
            "(structured outputs are incompatible with user tools there)"
        )
    return (
        f"response_format json_schema on non-native groq model "
        f"{request.model}: v1 serves its json_tool_call workaround (tools + "
        "forced tool_choice + json_mode + fake_stream — a cross-plane "
        "rewrite v2 deliberately does not reproduce); v1 owns it"
    )
