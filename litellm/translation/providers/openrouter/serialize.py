"""Serialize the IR into an openrouter ``/chat/completions`` request body.

v1's chain (httpx path, main.py:3354 dedicated elif): ``OpenrouterConfig.
map_openai_params`` = the base copy plus an always-set ``extra_body`` (its
transforms/models/route packing is DEAD at runtime — those kwargs ride the
top-level passthrough instead, openrouter not being in
``openai_compatible_providers``; the empty dict is popped back out by
``transform_request``), then ``transform_request`` = the cache_control
keep/move for cache-capable models (guard fallback), the base five-touch
assembly, the extra_body merge (a no-op for v2-visible shapes) and the
ALWAYS-injected ``"usage": {"include": true}``. The body is the
openai_compat assembly with three deltas:

- ``usage: {"include": true}`` injected unconditionally (v1's ``if "usage"
  not in response`` arm is unreachable for v2-sent shapes: no accepted
  param produces a ``usage`` body key).
- ``top_k`` emitted verbatim top-level (the non-compat passthrough,
  wire-proven in the request gate).
- ``reasoning_effort`` emitted verbatim (the gate already restricted it to
  reasoning-capable models, mirroring v1's dual-keyed supported list).

``max_completion_tokens`` passes VERBATIM (no rename arm). v1 keeps the wire
model bare on this path (no ``openrouter/`` prefix) — response/stream gates
pin it. The HTTP-Referer/X-Title headers and the litellm.OpenrouterConfig
class-attr extra_body merge live in completion()'s elif — envelope scope,
never here.
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
    return assemble_body(request).map(
        lambda body: _with_openrouter_deltas(body, request)
    )


def _with_openrouter_deltas(body: Body, request: ChatRequest) -> Body:
    extras: dict[str, PlainJson] = {"usage": {"include": True}}
    top_k = request.params.top_k.default_value(None)
    if top_k is not None:
        extras = {**extras, "top_k": top_k}
    effort = request.reasoning_effort.default_value(None)
    if effort is not None:
        extras = {**extras, "reasoning_effort": effort}
    return {**body, **extras}
