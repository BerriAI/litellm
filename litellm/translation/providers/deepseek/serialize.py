"""Serialize the IR into a deepseek ``/chat/completions`` request body.

v1's chain (httpx path, main.py:1942 dedicated elif): ``DeepSeekChatConfig.
map_openai_params`` = the base copy plus the thinking rewrite, then
``transform_request`` = the always-on content-list flatten + the inherited
base five-touch assembly (the thinking-mode history fill falls back in
params.py). The body is the openai_compat assembly with two deltas:

- text-only content lists are flattened to one concatenated string (the
  same ``handle_messages_with_content_list_to_str_conversion`` truth the
  compat_sdk flatten delta mirrors — IMPORTED, never copied; non-text
  shapes fell back at the gate so the helper's multimodal skip is inert
  here, exactly the sambanova configuration).
- the thinking rewrite (map_openai_params:49-63, verified in-process at
  HEAD): a verbatim ``thinking`` dict is accepted ONLY as
  ``{"type": "enabled"}`` (budget_tokens silently discarded, disabled and
  adaptive dropped — and a present-but-not-enabled dict SHADOWS
  reasoning_effort, the elif chain); otherwise ``reasoning_effort`` other
  than "none" becomes ``{"type": "enabled"}`` and the reasoning_effort key
  never reaches the wire. The rewrite is NOT model-gated.

``max_completion_tokens`` passes VERBATIM (no rename arm anywhere) and
``top_k`` falls back at the gate (extra_body packing, wire-proven in the
request gate). v1 keeps the wire model bare on this path (no ``deepseek/``
prefix) — response/stream gates pin it.
"""

from __future__ import annotations

from expression import Error, Result

from ...deps import TranslationDeps
from ...errors import TranslationError
from ...ir import Body, ChatRequest
from ..compat_sdk.serialize import flatten_text_lists
from ..openai_compat.serialize import assemble_body
from . import params as p

_SerializeResult = Result[Body, TranslationError]


def serialize_request(request: ChatRequest, deps: TranslationDeps) -> _SerializeResult:
    reason = p.unsupported_params(request, deps)
    if reason is not None:
        return Error(TranslationError.of_unsupported(reason))
    return assemble_body(request).map(lambda body: _with_deepseek_deltas(body, request))


def _with_deepseek_deltas(body: Body, request: ChatRequest) -> Body:
    flattened = flatten_text_lists(body)
    if not _thinking_enabled_after_map(request):
        return flattened
    return {**flattened, "thinking": {"type": "enabled"}}


def _thinking_enabled_after_map(request: ChatRequest) -> bool:
    return p.thinking_mode_requested(request)
