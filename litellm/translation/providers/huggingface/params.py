"""Parameter gates for the huggingface serializer (api_base route ONLY).

v1's ``HuggingFaceChatConfig.transform_request`` forks on
``litellm_params["api_base"]`` (chat/transformation.py:135):

- api_base SET (a dedicated endpoint / TGI server): the body is VERBATIM
  ``{model, messages, **optional_params}`` — no message transforms, no
  model rewrite, ``max_retries`` NOT popped. This is the ONE route v2
  ports (``deps.api_base`` carries the value; the fork must thread
  ``litellm_params["api_base"]`` into deps — CLAUDE.md HARD OBLIGATION).
- api_base UNSET (the router route): 3-segment ``provider/org/model``
  names trigger ``_fetch_inference_provider_mapping`` — a BLOCKING HTTP
  GET to the HF hub INSIDE the transform — and a providerId model
  rewrite; 2-segment names skip the fetch but run the base message
  transforms and the router URL synthesis; 1-segment names CRASH v1
  (ValueError on the split). I/O cannot live in a pure serializer and
  the route fans out per shape: the WHOLE router route falls back typed.

The param gate itself is the plain OpenAI base list (no map override):
mct passes VERBATIM, user is model-list gated (silently dropped), and
top_k rides the TOP-LEVEL passthrough (huggingface is NOT in
openai_compatible_providers) — v1 SERVES it, so v2 emits it verbatim.
"""

from __future__ import annotations

from ...deps import TranslationDeps
from ...ir import ChatRequest
from ..compat_sdk.checks import BASE_LIST, unsupported_against, user_note
from ..openai_compat.params import unsupported_response_format

_HUGGINGFACE_LIST = BASE_LIST | frozenset({"top_k"})


def unsupported_params(request: ChatRequest, deps: TranslationDeps) -> str | None:
    if deps.api_base is None:
        return (
            "huggingface router route (no api_base): v1 synthesizes router."
            "huggingface.co URLs and 3-segment names fetch the HF inference-"
            "provider mapping over HTTP inside transform_request; only the "
            "api_base route is ported — v1 handles the router route"
        )
    return unsupported_against(
        request,
        provider="huggingface",
        allowed=_HUGGINGFACE_LIST,
        notes={"user": user_note("huggingface")},
    ) or unsupported_response_format(request)
