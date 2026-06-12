"""Serialize the IR into a Mistral ``/v1/chat/completions`` request body.

v1's chain is ``MistralConfig.map_openai_params`` (explicit arms, no
supported-list loop) then ``transform_request`` -> ``_transform_messages``
(probed in-process at HEAD). The body is the openai_compat assembly with:

- ``max_completion_tokens`` -> ``max_tokens`` (mct wins over max_tokens);
- ``tool_choice``: strings only — ``required`` -> ``any`` (auto/none pass);
  a dict tool_choice is SILENTLY DROPPED by v1's ``isinstance(value, str)``
  arm, so the IR's ``specific`` tag is dropped here too (served delta);
- tools: ``$id``/``$schema`` stripped recursively with v1's EXACT depth cap
  — ``_remove_json_schema_refs(max_depth=DEFAULT_MAX_RECURSE_DEPTH)``
  (= 100 at HEAD; the ``max_depth=10`` signature default is dead at this
  call site, verifier-wave2b-beta F2). Levels deeper than the cap keep
  their keys; additionalProperties/strict are KEPT, the v1 docstring
  notwithstanding. v2 imports the SAME ``litellm.constants`` symbol v1
  reads (the allowed env-seeded leaf), so the env-overridable cap can
  never drift between the two sides;
- ``top_k`` emitted verbatim top-level (the generic passthrough);
- ``stream`` only when True (assemble_body already matches v1's
  value-is-True arm).

Message munging mirrors ``_transform_messages``' two branches:

- ANY image content block in the request -> the base-transform branch:
  openai_compat's messages ride verbatim (no flatten, no name drop, no
  tool-call rebuild, no empty-assistant removal; ``file`` parts fall back
  at the inbound parse before reaching here);
- otherwise: text content lists FLATTEN to one string (v1's
  ``convert_content_list_to_str`` only assigns a TRUTHY join, so a list
  flattening to "" keeps its list form in v1 — typed fallback), tool_calls
  are already in the MistralToolCallMessage shape (id/type/function — the
  IR admits nothing else), EMPTY assistant messages are REMOVED (every
  key None/absent except role, content "" included), and None values are
  stripped from every message (v1's extra_forbidden prevention).
"""

from __future__ import annotations

from expression import Error, Ok, Result

from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH

from ...deps import TranslationDeps
from ...errors import TranslationError
from ...ir import Body, ChatRequest, PlainJson
from ..openai_compat.serialize import assemble_body
from . import params as p

_SerializeResult = Result[Body, TranslationError]


def serialize_request(request: ChatRequest, deps: TranslationDeps) -> _SerializeResult:
    reason = p.unsupported_params(request)
    if reason is not None:
        return Error(TranslationError.of_unsupported(reason))
    return assemble_body(request).bind(lambda body: _with_mistral_deltas(body, request))


def _with_mistral_deltas(body: Body, request: ChatRequest) -> _SerializeResult:
    reshaped: dict[str, PlainJson] = {}
    for key, value in body.items():
        if key == "max_completion_tokens":
            reshaped = {**reshaped, "max_tokens": value}
        elif key == "tool_choice":
            choice = _mapped_tool_choice(value)
            if choice is not None:
                reshaped = {**reshaped, "tool_choice": choice}
        elif key == "tools":
            reshaped = {**reshaped, "tools": _without_schema_refs(value, 0)}
        else:
            reshaped = {**reshaped, key: value}
    top_k = request.params.top_k.default_value(None)
    if top_k is not None:
        reshaped = {**reshaped, "top_k": top_k}
    messages = _mistral_messages(reshaped.get("messages"), request)
    if isinstance(messages, TranslationError):
        return Error(messages)
    return Ok({**reshaped, "messages": messages})


def _mapped_tool_choice(value: PlainJson) -> PlainJson | None:
    """v1 ``_map_tool_choice`` behind an ``isinstance(value, str)`` gate: a
    dict tool_choice (the IR's ``specific`` tag) is silently dropped;
    ``required`` -> ``any``; auto/none pass (any other string also maps to
    ``any``, but the inbound parse admits only the four canonical tags)."""
    if not isinstance(value, str):
        return None
    return "any" if value == "required" else value


def _without_schema_refs(value: PlainJson, depth: int) -> PlainJson:
    """Mirror v1's ``_remove_json_schema_refs(max_depth=
    DEFAULT_MAX_RECURSE_DEPTH)`` exactly: v1 counts max_depth DOWN from the
    constant starting at the tools list and strips while ``max_depth > 0``;
    v2 counts UP from 0 at the same list, so a dict at v2-depth ``d`` is
    stripped iff ``d < DEFAULT_MAX_RECURSE_DEPTH`` — the same node set.
    DEEPER levels keep their keys (the cap returns the subtree untouched)."""
    if depth >= DEFAULT_MAX_RECURSE_DEPTH:
        return value
    if isinstance(value, dict):
        return {
            key: _without_schema_refs(item, depth + 1)
            for key, item in value.items()
            if key not in ("$id", "$schema")
        }
    if isinstance(value, list):
        return [_without_schema_refs(item, depth + 1) for item in value]
    return value


def _mistral_messages(
    messages: PlainJson, request: ChatRequest
) -> list[PlainJson] | TranslationError:
    if not isinstance(messages, list):
        return TranslationError.of_unsupported(
            "mistral serializer received no message list; wiring bug"
        )
    if _request_has_image(request):
        return messages  # v1's base-transform branch: verbatim passthrough
    reshaped: list[PlainJson] = []
    for message in messages:
        if not isinstance(message, dict):
            return TranslationError.of_unsupported(
                "mistral serializer received a non-object message; wiring bug"
            )
        flattened = _flatten_content(message)
        if isinstance(flattened, TranslationError):
            return flattened
        if _is_empty_assistant(flattened):
            continue
        reshaped = [
            *reshaped,
            {key: value for key, value in flattened.items() if value is not None},
        ]
    return reshaped


def _request_has_image(request: ChatRequest) -> bool:
    return any(
        block.tag == "image"
        for message in request.messages
        for block in message.content
    )


def _flatten_content(
    message: dict[str, PlainJson],
) -> dict[str, PlainJson] | TranslationError:
    content = message.get("content")
    if not isinstance(content, list):
        return message
    texts = "".join(
        text
        for item in content
        if isinstance(item, dict) and isinstance(text := item.get("text"), str)
    )
    if not texts:
        return TranslationError.of_unsupported(
            "text content list flattening to an empty string: v1's truthy "
            "assignment keeps the LIST form on the wire; v1 serves it"
        )
    return {**message, "content": texts}


def _is_empty_assistant(message: dict[str, PlainJson]) -> bool:
    """v1 ``_is_empty_assistant_message``: every assistant-message key is
    None/absent except role, with content ``""`` counting as empty."""
    if message.get("role") != "assistant":
        return False
    return all(
        value is None or (key == "content" and value == "")
        for key, value in message.items()
        if key != "role"
    )
