"""Parameter gates for the sagemaker_chat serializer.

v1's gate is ``_check_valid_arg`` over the BASE OpenAI GPT supported list
(no override — the widest wave-2b list). Probed in-process at HEAD:

- ``thinking``/``reasoning_effort`` RAISE (outside the base list);
- ``user`` is silently dropped (model-list gated upstream);
- ``response_format`` RAISES on endpoint names literally ``gpt-4`` /
  ``gpt-3.5-turbo-16k`` (the base-list name gate — SageMaker endpoint
  names are arbitrary, so the corner is reachable);
- ``max_completion_tokens`` passes VERBATIM (no rename — the widest-list
  exception in the wave) and ``top_k`` rides top-level via the generic
  passthrough (both served);
- n/seed/penalties/logprobs/logit_bias/web_search_options pass through in
  v1 but are parse-level unknowns in v2 (typed fallback, v1 serves).
"""

from __future__ import annotations

from ...ir import ChatRequest

_RESPONSE_FORMAT_RAISE_NAMES = ("gpt-4", "gpt-3.5-turbo-16k")


def unsupported_params(request: ChatRequest) -> str | None:
    if request.thinking.is_some():
        return "thinking is not a sagemaker_chat param; v1 raises or drops it"
    if request.reasoning_effort.is_some():
        return "reasoning_effort is not a sagemaker_chat param; v1 raises or drops it"
    if request.user.is_some():
        return (
            "user is model-list gated upstream and silently dropped for "
            "sagemaker_chat; v1 serves its own drop"
        )
    if (
        request.response_format.is_some()
        and request.model in _RESPONSE_FORMAT_RAISE_NAMES
    ):
        return (
            f"response_format on an endpoint literally named {request.model}: "
            "v1's base-list name gate raises UnsupportedParamsError"
        )
    return None
