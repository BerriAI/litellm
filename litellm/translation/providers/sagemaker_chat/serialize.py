"""Serialize the IR into a SageMaker Messages-API ``/invocations`` body.

v1's chain is the BASE ``OpenAIGPTConfig.map_openai_params`` +
``transform_request`` with no overrides: the body IS the openai_compat
assembly (the model field carries the SageMaker ENDPOINT NAME — it also
routes the URL, envelope scope), plus ``top_k`` riding the generic
passthrough top-level (wire-proven) and ``max_completion_tokens`` verbatim
(no rename — assemble_body already re-emits the caller's key). SigV4 and
the ``/invocations[-response-stream]`` URL split are envelope; the
``supports_stream_param_in_request_body = False`` property only governs
the handler's stream-key ADD for streaming sends — the serialized body
mirrors v1's transform output exactly.
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
    reason = p.unsupported_params(request)
    if reason is not None:
        return Error(TranslationError.of_unsupported(reason))
    return assemble_body(request).map(lambda body: _with_top_k(body, request))


def _with_top_k(body: Body, request: ChatRequest) -> Body:
    top_k = request.params.top_k.default_value(None)
    if top_k is None:
        return body
    return {**body, "top_k": top_k}
