"""Parameter gates for the mistral chat serializer.

v1's gate is ``_check_valid_arg`` over ``MistralConfig.
get_supported_openai_params`` (stream/temperature/top_p/max_tokens/mct/
tools/tool_choice/seed/stop/response_format/parallel_tool_calls; +
thinking/reasoning_effort iff "magistral" in the model). Everything the IR
carries that v1 raises or rewrites out-of-band falls back typed so v1
serves its own behavior (probed in-process at HEAD):

- ``user``: SILENTLY DROPPED upstream (dossier drift — researcher-4 listed
  it as a raise; the probe shows the model-list special case eats it) —
  typed fallback, v1 serves the drop.
- ``thinking``/``reasoning_effort``: on magistral models v1 sets an internal
  ``_add_reasoning_prompt`` flag and INJECTS a reasoning system prompt at
  transform time (a body rewrite v2 deliberately does not reproduce); on
  every other model v1 raises — one fallback arm covers both, v1 serves.
- ``top_k`` is NOT a raise (second dossier drift): the generic passthrough
  places it top-level in optional_params and the body carries it verbatim —
  SERVED by the serializer, wire-proven in the request gate.
- n/seed/penalties/logprobs are parse-level unknowns (typed fallback at the
  inbound boundary; v1 serves seed via ``extra_body.random_seed``).
"""

from __future__ import annotations

from ...ir import ChatRequest


def unsupported_params(request: ChatRequest) -> str | None:
    if request.user.is_some():
        return (
            "user is model-list gated upstream and silently dropped for "
            "mistral; v1 serves its own drop"
        )
    if request.thinking.is_some():
        return (
            "thinking on mistral: v1 injects a reasoning system prompt on "
            "magistral models (a transform-time body rewrite) and raises "
            "UnsupportedParamsError elsewhere; v1 owns both"
        )
    if request.reasoning_effort.is_some():
        return (
            "reasoning_effort on mistral: v1 injects a reasoning system "
            "prompt on magistral models (a transform-time body rewrite) and "
            "raises UnsupportedParamsError elsewhere; v1 owns both"
        )
    return None
