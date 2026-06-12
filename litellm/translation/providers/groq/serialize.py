"""Serialize the IR into a groq ``/openai/v1/chat/completions`` request body.

v1's chain is ``GroqChatConfig.map_openai_params`` -> ``transform_request``
(the assistant None-strip pre-step, then the base GPT assembly) on the
dedicated httpx elif (probed in-process at HEAD). The body is the
openai_compat assembly with:

- ``max_completion_tokens`` -> ``max_tokens`` (groq's map forwards to the
  OpenAILike rename positionally, so the rename DOES run — its own
  signature's ``replace_...=False`` default is never threaded);
- ``reasoning_effort`` emitted verbatim on capability-flagged models (the
  params gate already fell back elsewhere);
- ``top_k`` top-level: v1 packs it into ``extra_body`` and hh merges
  extra_body into the wire body — wire-equivalent emission (the gate
  mirrors hh's merge);
- assistant messages rebuilt WITHOUT None-valued keys (issue #5839 —
  ``content: None`` on tool-call messages never reaches the wire);
- ``response_format`` verbatim (json_object always; json_schema only on
  native-schema models — the params gate owns the other arms). The
  fake_stream/json_mode keys v1's map sets are ROUTING keys hh pops
  before the wire; they never appear here.
"""

from __future__ import annotations

from expression import Error, Result

from ...deps import TranslationDeps
from ...errors import TranslationError
from ...ir import Body, ChatRequest, PlainJson
from ..openai_compat.serialize import assemble_body
from . import params as p

_SerializeResult = Result[Body, TranslationError]


def serialize_request(request: ChatRequest, deps: TranslationDeps) -> _SerializeResult:
    reason = p.unsupported_params(request, deps)
    if reason is not None:
        return Error(TranslationError.of_unsupported(reason))
    return assemble_body(request).map(lambda body: _with_groq_deltas(body, request))


def _with_groq_deltas(body: Body, request: ChatRequest) -> Body:
    reshaped: dict[str, PlainJson] = {}
    for key, value in body.items():
        if key == "max_completion_tokens":
            reshaped = {**reshaped, "max_tokens": value}
        elif key == "messages" and isinstance(value, list):
            reshaped = {
                **reshaped,
                "messages": [_without_assistant_nones(message) for message in value],
            }
        else:
            reshaped = {**reshaped, key: value}
    effort = request.reasoning_effort.default_value(None)
    if effort is not None:
        reshaped = {**reshaped, "reasoning_effort": effort}
    top_k = request.params.top_k.default_value(None)
    if top_k is not None:
        reshaped = {**reshaped, "top_k": top_k}
    return reshaped


def _without_assistant_nones(message: PlainJson) -> PlainJson:
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return message
    return {key: value for key, value in message.items() if value is not None}
