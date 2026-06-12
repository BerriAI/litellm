"""Parameter gates for the watsonx chat serializer.

v1's gate is ``_check_valid_arg`` over ``IBMWatsonXChatConfig.
get_supported_openai_params`` (temperature/max_tokens/top_p/
frequency_penalty/stop/seed/stream/tools/tool_choice/logprobs/top_logprobs/
n/presence_penalty/response_format/reasoning_effort) PLUS the watsonx-only
extra arm in get_optional_params: any legacy watsonx-TEXT param (top_k,
decoding_method, ...) raises a bare ValueError telling you to use the
``watsonx_text`` provider. Probed in-process at HEAD:

- ``max_completion_tokens`` RAISES (NOT renamed — the OpenAILike rename is
  dead behind the list gate);
- ``top_k`` hits the legacy-text ValueError (a DIFFERENT raise class than
  UnsupportedParamsError — pinned in the request gate);
- ``parallel_tool_calls``/``thinking`` RAISE; ``user`` is silently dropped;
- ``reasoning_effort`` is served VERBATIM (unconditionally in the list);
- penalties/n/seed/logprobs/top_logprobs are served by v1 but are
  parse-level unknowns in v2 (typed fallback, v1 serves).

Two payload gates live here too (probed):

- ``deployment/`` models: v1 routes them to deployment URLs, POPS
  ``api_version`` from optional_params into the URL, and injects NO
  model_id/project_id — envelope behavior v2 does not reproduce;
- missing project AND space id in deps: v1's ``_get_api_params`` raises
  WatsonXAIError 401 — fall back so v1 serves its own raise.
"""

from __future__ import annotations

from ...deps import TranslationDeps
from ...ir import ChatRequest


def unsupported_params(request: ChatRequest, deps: TranslationDeps) -> str | None:
    if request.model.startswith("deployment/"):
        return (
            "watsonx deployment/ model: v1 routes it to the deployment URL "
            "with no model_id/project_id payload and pops api_version into "
            "the URL (envelope); v1 owns it"
        )
    if deps.watsonx_project_id is None and deps.watsonx_space_id is None:
        return (
            "watsonx project_id/space_id unresolved: v1's _get_api_params "
            "raises WatsonXAIError 401; v1 serves its own raise"
        )
    if request.params.max_completion_tokens.is_some():
        return (
            "max_completion_tokens is outside watsonx's supported list; "
            "v1's get_optional_params raises UnsupportedParamsError (the "
            "OpenAILike rename arm is dead behind the list gate)"
        )
    if request.params.top_k.is_some():
        return (
            "top_k is a legacy watsonx-text param: v1's get_optional_params "
            "raises ValueError pointing at the watsonx_text provider"
        )
    if request.parallel_tool_calls.is_some():
        return (
            "parallel_tool_calls is outside watsonx's supported list; "
            "v1 raises or drops it"
        )
    if request.thinking.is_some():
        return "thinking is not a watsonx chat param; v1 raises or drops it"
    if request.user.is_some():
        return (
            "user is model-list gated upstream and silently dropped for "
            "watsonx; v1 serves its own drop"
        )
    return None
