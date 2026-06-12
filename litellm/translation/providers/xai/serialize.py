"""Serialize the IR into an xai (Grok) ``/v1/chat/completions`` request body.

v1's chain is ``XAIChatConfig.map_openai_params`` (own supported list, tools
``strict`` strip) then ``transform_request`` = ``strip_name_from_messages``
+ the inherited OpenAIGPTConfig five-touch assembly. The body is therefore
the openai_compat assembly with three deltas: the function-level ``strict``
key is stripped from every tool (v1 ``filter_value_from_dict(tool,
"strict")``; deeper ``strict`` keys fall back in the raw guard), and ``user``
/ ``reasoning_effort`` — typed fallbacks on plain GPT — are emitted verbatim
because xai supports them (reasoning_effort gated per model in params.py).
The non-user message ``name`` strip needs no code: the IR never carries
``name`` and the xai raw guard only falls back on user-message names.
"""

from __future__ import annotations

from expression import Error, Result

from ...deps import TranslationDeps
from ...errors import TranslationError
from ...ir import Body, ChatRequest, PlainJson
from ..openai_compat.serialize import assemble_body, strip_function_strict
from . import params as p

_SerializeResult = Result[Body, TranslationError]


def serialize_request(request: ChatRequest, deps: TranslationDeps) -> _SerializeResult:
    reason = p.unsupported_params(request, deps)
    if reason is not None:
        return Error(TranslationError.of_unsupported(reason))
    return assemble_body(request).map(lambda body: _with_xai_deltas(body, request))


def _with_xai_deltas(body: Body, request: ChatRequest) -> Body:
    extras: dict[str, PlainJson] = {}
    user = request.user.default_value(None)
    if user is not None:
        extras = {**extras, "user": user}
    effort = request.reasoning_effort.default_value(None)
    if effort is not None:
        extras = {**extras, "reasoning_effort": effort}
    tools = body.get("tools")
    if not isinstance(tools, list):
        return {**body, **extras}
    # the function-level strict strip is openai_compat.serialize.
    # strip_function_strict (lifted there when fireworks_ai became the
    # second consumer)
    return {**body, "tools": [strip_function_strict(tool) for tool in tools], **extras}
