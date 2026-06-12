"""Serialize the IR into a hosted_vllm ``/chat/completions`` request body.

v1's chain (httpx path, main.py:2619 dedicated elif): ``HostedVLLMChatConfig.
map_openai_params`` = tools cleaning + the thinking->reasoning_effort budget
rewrite, then the base copy; ``_transform_messages`` handles shapes that
fall back at the shared boundaries (params.py). The body is the
openai_compat assembly with three deltas (all verified in-process at HEAD):

- tools cleaned: ``additionalProperties: false`` removed at EVERY depth
  (only the ``false`` value — v1's ``_remove_additional_properties``) and
  ``strict`` removed at EVERY depth (v1's ``_remove_strict_from_schema``;
  contrast xai, which strips function-level strict only and falls back on
  deeper keys).
- ``thinking`` rewritten to ``reasoning_effort`` by v1's budget bands —
  enabled: >=10000 high / >=5000 medium / >=2000 low / else (incl. absent)
  minimal; disabled and adaptive DROPPED; an explicit ``reasoning_effort``
  WINS over thinking (v1's "not already set" arm). The thinking key never
  reaches the wire.
- ``reasoning_effort`` emitted verbatim (unconditionally supported).

``max_completion_tokens`` passes VERBATIM and ``top_k`` falls back at the
gate (hosted_vllm IS in openai_compatible_providers: extra_body packing,
wire-proven in the request gate). v1 keeps the wire model bare on this path
(no ``hosted_vllm/`` prefix) — response/stream gates pin it.
"""

from __future__ import annotations

from ...ir import Body, ChatRequest, PlainJson, ThinkingParam
from ..openai_compat.serialize import make_gated_serializer
from . import params as p


def _with_hosted_vllm_deltas(body: Body, request: ChatRequest) -> Body:
    extras: dict[str, PlainJson] = {}
    effort = _effective_reasoning_effort(request)
    if effort is not None:
        extras = {**extras, "reasoning_effort": effort}
    tools = body.get("tools")
    if isinstance(tools, list):
        return {**body, "tools": [_cleaned(tool) for tool in tools], **extras}
    return {**body, **extras}


def _effective_reasoning_effort(request: ChatRequest) -> PlainJson:
    effort = request.reasoning_effort.default_value(None)
    if effort is not None:
        return effort
    thinking = request.thinking.default_value(None)
    if thinking is None:
        return None
    return _band(thinking)


def _band(thinking: ThinkingParam) -> str | None:
    if thinking.tag != "enabled":
        return None  # v1 pops disabled/adaptive without a rewrite
    budget = thinking.enabled.budget_tokens.default_value(0)
    if budget >= 10000:
        return "high"
    if budget >= 5000:
        return "medium"
    if budget >= 2000:
        return "low"
    return "minimal"


def _cleaned(value: PlainJson) -> PlainJson:
    """v1's two recursive walks in one pass: drop ``strict`` (any value) and
    ``additionalProperties`` ONLY when false, at every depth."""
    if isinstance(value, dict):
        return {
            key: _cleaned(item)
            for key, item in value.items()
            if key != "strict" and not (key == "additionalProperties" and item is False)
        }
    if isinstance(value, list):
        return [_cleaned(item) for item in value]
    return value


serialize_request = make_gated_serializer(p.unsupported_params, _with_hosted_vllm_deltas)
