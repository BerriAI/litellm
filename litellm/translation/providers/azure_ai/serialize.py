"""Serialize the IR for the azure_ai (Azure AI Foundry) openai-compat route.

v1 serves this route over httpx with ``AzureAIStudioConfig(OpenAIConfig)``:
``map_openai_params`` is the plain GPT mapping (OpenAIConfig delegates per
model family) and ``transform_request`` is ``{model, messages,
**optional_params}`` after an ``extra_body`` merge and ``max_retries`` pop --
both unreachable here (the inbound schema forbids extra_body; max_retries is
an SDK kwarg) -- so body assembly is the openai_compat serializer verbatim.
The azure_ai deltas are gates: the content-list flatten lives in the raw
guard; ``tool_choice`` support is a per-model model-map filter in v1
(``supports_tool_choice("azure_ai/{model}")``, transformation.py:32-42) and
fails closed; grok models get a stop-param filter via ``XAIChatConfig``;
o-series/gpt-5/gpt-audio names dispatch onto unported param families. The
hostname-keyed api-key-vs-Bearer auth, the ``/models/chat/completions`` URL
and the api-version query are envelope. The response-side ``azure_ai/{model}``
rename lives in this package's response module.
"""

from __future__ import annotations

from expression import Error, Result

from ...deps import TranslationDeps
from ...errors import TranslationError
from ...ir import Body, ChatRequest
from ..openai_compat import params as openai_params
from ..openai_compat import serialize_request as openai_compat_serialize_request

_SerializeResult = Result[Body, TranslationError]


def serialize_request(request: ChatRequest, deps: TranslationDeps) -> _SerializeResult:
    reason = (
        _unsupported_model(request.model)
        or openai_params.unsupported_params(request)
        or _unsupported_tool_choice(request)
    )
    if reason is not None:
        return Error(TranslationError.of_unsupported(reason))
    return openai_compat_serialize_request(request, deps)


def _unsupported_model(model: str) -> str | None:
    if "grok" in model:
        return (
            f"grok model {model} on azure_ai: v1 filters its params through "
            "XAIChatConfig (azure_ai/chat/transformation.py:50-58)"
        )
    if "audio" in model:
        return (
            f"audio model {model}: v1's OpenAIGPTAudioConfig may own its "
            "param mapping (openai/openai.py:232-241)"
        )
    return openai_params.unsupported_model_family(model)


def _unsupported_tool_choice(request: ChatRequest) -> str | None:
    if request.tool_choice.is_none():
        return None
    return (
        "tool_choice on azure_ai is model-map gated in v1 "
        "(supports_tool_choice, azure_ai/chat/transformation.py:32-42); v1 handles it"
    )
