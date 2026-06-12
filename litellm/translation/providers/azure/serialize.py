"""Serialize the IR into an Azure OpenAI chat-completions request body.

v1's azure body is the same five-touch openai passthrough
(``AzureOpenAIConfig.transform_request`` runs ``convert_to_azure_openai_messages``
-- a no-op on the admitted surface: images are already objects and legacy
``function_call`` fails closed at parse -- then ``{model, messages,
**optional_params}``). The azure deltas are gates, not body shape: the
api-version legality of ``tool_choice``/``response_format``, family detection
on ``base_model or model``, and the absence of the cache_control strip (the
azure raw guard rejects cache_control instead, so the shared serializer's
strip is a no-op here). Body assembly is the openai_compat serializer
verbatim; its own openai-only gates (exact-name response_format models,
prefix-form family checks on the deployment name) can only WIDEN the
fallback surface, never change a served body. The deployment name stays the
body ``model``; the api-version URL query and api-key/AD-token auth are
envelope (translation_seam + Endpoint).
"""

from __future__ import annotations

from expression import Error, Result

from ...deps import TranslationDeps
from ...errors import TranslationError
from ...ir import Body, ChatRequest
from ..openai_compat import serialize_request as openai_compat_serialize_request
from ..openai_compat import params as openai_params
from . import params as p

_SerializeResult = Result[Body, TranslationError]


def serialize_request(request: ChatRequest, deps: TranslationDeps) -> _SerializeResult:
    reason = (
        p.unsupported_model_family(request.model, deps)
        or openai_params.unsupported_params(request)
        or p.unsupported_tool_choice(request, deps)
        or p.unsupported_response_format(request, deps)
    )
    if reason is not None:
        return Error(TranslationError.of_unsupported(reason))
    return openai_compat_serialize_request(request, deps)
