"""Serialize the IR into a Cohere v2 ``/v2/chat`` request body.

v1's chain is ``CohereV2ChatConfig.map_openai_params`` (rename arms) then
the inherited OpenAI GPT ``transform_request`` — the wire body is the
openai_compat assembly with the cohere renames applied (verified in-process
at HEAD):

- ``top_p`` -> ``p``; ``stop`` -> ``stop_sequences``;
- ``max_completion_tokens`` -> ``max_tokens`` (mct wins; the raw guard
  rejects requests carrying both keys);
- ``tool_choice`` DROPPED silently (in v1's supported list, no map arm);
- ``top_k`` emitted verbatim (v1's generic non-openai passthrough places it
  top-level in optional_params — dossier drift, re-verified at HEAD);
- ``n``/``seed``/penalties are served by v1 but are not IR params: they fall
  back at the inbound parse, so v1 serves them (typed fallback rows).
"""

from __future__ import annotations

from ...ir import Body, ChatRequest, PlainJson
from ..openai_compat.serialize import make_gated_serializer
from . import params as p

_RENAMES = (
    ("top_p", "p"),
    ("stop", "stop_sequences"),
    ("max_completion_tokens", "max_tokens"),
)




def _with_cohere_deltas(body: Body, request: ChatRequest) -> Body:
    renamed: dict[str, PlainJson] = {
        _renamed_key(key): value
        for key, value in body.items()
        # v1 silently drops tool_choice (supported list, no map arm)
        if key != "tool_choice"
    }
    top_k = request.params.top_k.default_value(None)
    if top_k is not None:
        renamed = {**renamed, "top_k": top_k}
    return renamed


def _renamed_key(key: str) -> str:
    for original, wire in _RENAMES:
        if key == original:
            return wire
    return key


serialize_request = make_gated_serializer(p.unsupported_params, _with_cohere_deltas)
