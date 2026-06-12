"""Serialize the IR into a huggingface dedicated-endpoint request body.

The api_base route's v1 body is VERBATIM ``ChatCompletionRequest(model,
messages, **optional_params)`` — the request model untouched, messages
untouched (no flatten, no base image transforms; the openai guard's
fallbacks are all the v1-serves kind), no renames anywhere
(``max_completion_tokens`` passes verbatim). The openai_compat assembly
reproduces it, plus ``top_k`` verbatim (the non-compat top-level
passthrough, wire-proven in the request gate). The router route never
reaches here (params.py falls back on it). v1 keeps the wire model bare
on this path — response/stream gates pin it.
"""

from __future__ import annotations

from expression import Error, Result

from ...deps import TranslationDeps
from ...errors import TranslationError
from ...ir import Body, ChatRequest
from ..openai_compat.serialize import assemble_body
from . import params as p

_SerializeResult = Result[Body, TranslationError]


def serialize_request(request: ChatRequest, deps: TranslationDeps) -> _SerializeResult:
    reason = p.unsupported_params(request, deps)
    if reason is not None:
        return Error(TranslationError.of_unsupported(reason))
    return assemble_body(request).map(
        lambda body: _with_huggingface_deltas(body, request)
    )


def _with_huggingface_deltas(body: Body, request: ChatRequest) -> Body:
    top_k = request.params.top_k.default_value(None)
    if top_k is None:
        return body
    return {**body, "top_k": top_k}
