"""Serialize the IR into a fireworks_ai ``/chat/completions`` request body.

v1's chain (httpx path, main.py:2198 dedicated elif): the EXPLICIT-arm
``map_openai_params`` (tool_choice required->any, mct->max_tokens rename,
copy-if-supported) then ``transform_request`` = the model rewrite + the
message/tools helper + the base five-touch assembly. The body is the
openai_compat assembly with these deltas (all verified in-process at HEAD):

- model rewritten to ``accounts/fireworks/models/{model}`` UNLESS it
  already starts with ``accounts/`` or contains ``#`` (deployment ids).
- ``max_completion_tokens`` -> ``max_tokens`` (fireworks HAS the rename
  arm, unlike its wave-2b siblings).
- ``tool_choice`` ``"required"`` -> ``"any"``.
- tools: the FUNCTION-LEVEL ``strict`` pop (the shared
  ``strip_function_strict``; deeper strict keys ride verbatim — v1 keeps
  them too, contrast hosted_vllm). Legacy-def unpacking fell back at the
  gate, so the serializer never sees those schemas.
- image ``#transform=inline`` suffix appended to every non-``data:``
  image url when ``"vision"`` is NOT a substring of the REWRITTEN model
  (the literal check — VL models without "vision" in the name get the
  suffix too, pinned). The ambient
  ``litellm.disable_add_transform_inline_image_block`` global and the
  per-request litellm_params flag are SEAM scope: the fork must force v1
  when either is set (CLAUDE.md HARD OBLIGATION) — this serializer encodes
  the default (enabled) arm.
- ``user`` and capability-gated ``reasoning_effort`` emitted verbatim
  (both in v1's own list).

v1 also strips ``cache_control`` recursively (== the IR drop, served) and
pops ``provider_specific_fields``/``thinking_blocks`` from messages (both
fall back at the shared boundaries; v1 serves the pops). The response
model is ``fireworks_ai/{WIRE model}`` — response.py owns that prefix.
"""

from __future__ import annotations

from ...ir import Body, ChatRequest, PlainJson
from ..openai_compat.serialize import make_gated_serializer, strip_function_strict
from . import params as p

_MODEL_PREFIX = "accounts/fireworks/models/"


def _wire_model(model: str) -> str:
    if model.startswith("accounts/") or "#" in model:
        return model
    return f"{_MODEL_PREFIX}{model}"


def _with_fireworks_deltas(body: Body, request: ChatRequest) -> Body:
    model = _wire_model(request.model)
    patched: dict[str, PlainJson] = {**body, "model": model}
    if "max_completion_tokens" in patched:
        patched = {
            key: value
            for key, value in patched.items()
            if key != "max_completion_tokens"
        } | {"max_tokens": patched["max_completion_tokens"]}
    if patched.get("tool_choice") == "required":
        patched = {**patched, "tool_choice": "any"}
    tools = patched.get("tools")
    if isinstance(tools, list):
        patched = {**patched, "tools": [strip_function_strict(tool) for tool in tools]}
    messages = patched.get("messages")
    if isinstance(messages, list) and "vision" not in model:
        patched = {
            **patched,
            "messages": [_with_inline_images(message) for message in messages],
        }
    user = request.user.default_value(None)
    if user is not None:
        patched = {**patched, "user": user}
    effort = request.reasoning_effort.default_value(None)
    if effort is not None:
        patched = {**patched, "reasoning_effort": effort}
    return patched


def _with_inline_images(message: PlainJson) -> PlainJson:
    if not isinstance(message, dict) or message.get("role") != "user":
        return message
    content = message.get("content")
    if not isinstance(content, list):
        return message
    return {**message, "content": [_with_inline_suffix(part) for part in content]}


def _with_inline_suffix(part: PlainJson) -> PlainJson:
    if not isinstance(part, dict) or part.get("type") != "image_url":
        return part
    image = part.get("image_url")
    if not isinstance(image, dict):
        return part
    url = image.get("url")
    if not isinstance(url, str) or url.lower().startswith("data:"):
        return part
    return {**part, "image_url": {**image, "url": f"{url}#transform=inline"}}


serialize_request = make_gated_serializer(p.unsupported_params, _with_fireworks_deltas)
