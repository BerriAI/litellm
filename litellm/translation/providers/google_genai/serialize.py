"""Serialize the IR onto the generateContent wire format.

ONE serializer family for both google routes; the ``target`` parameter is the
researcher-2 drift list and nothing else: AI Studio refuses https media (v1
downloads it) and forwards function-call ids on gemini-3+; vertex passes
https media through as ``file_data`` and would attach ``labels`` (not
expressible in the IR). top_k rides BOTH routes (it is a provider kwarg, not
a gated OpenAI param — verified against v1 in-process). Auth, hosts, and API
versions are envelope and never appear here.

Fail-closed shapes (each names the v1 path): cache markers that could reach
``check_and_create_cache``'s network call, gemini-3 thinking budgets (read an
ambient litellm global), reasoning_effort xhigh/max (v1 raises), and anything
the message/tool converters decline.
"""

from __future__ import annotations

import copy
import json

from expression import Error, Ok, Option, Result
from expression.collections import Block
from typing_extensions import assert_never

from ...deps import TranslationDeps
from ...errors import TranslationError
from ...ir import Body, ChatRequest, ContentBlock, Message, PlainJson
from . import params as p
from .messages import serialize_contents, serialize_system, with_schema_prompt
from .schema import build_vertex_schema, strip_additional_properties, strip_strict
from .tools import serialize_tools, tool_config

_SerializeResult = Result[Body, TranslationError]

# Gemini's context-cache minimum is 1024 TOKENS (is_prompt_caching_valid_prompt
# runs token_counter over the continuous block from the FIRST to the LAST
# cache-marked message, unmarked messages in between included, and v1 makes
# the cachedContents network call at >= 1024). v2 must prove that call
# unreachable, so the gate computes a conservative UPPER bound on whatever v1
# could count; over-counting only widens the typed fallback:
#
# - a BPE token spans at least one BYTE, so UTF-8 byte length bounds the
#   token count of any text (char length does NOT: CJK/emoji code points are
#   multi-byte and can be multi-token);
# - v1 charges DEFAULT_IMAGE_TOKEN_COUNT (250) per image at zero text bytes,
#   and an unmarked image can sit inside v1's continuous block, so ANY media
#   block anywhere in a marker-bearing request fails closed;
# - token_counter adds per-message overhead and counts tool_use argument
#   JSON, so argument/name bytes are folded in and every system entry and
#   message contributes a fixed margin;
# - the byte total spans the WHOLE request (system + all messages), a strict
#   superset of v1's continuous cached block.
_CACHE_MARKER_TOKEN_LIMIT = 1024
_PER_MESSAGE_TOKEN_MARGIN = 8


def serialize_request_vertex(
    request: ChatRequest, deps: TranslationDeps
) -> _SerializeResult:
    return _serialize(request, deps, "vertex_ai")


def serialize_request_studio(
    request: ChatRequest, deps: TranslationDeps
) -> _SerializeResult:
    return _serialize(request, deps, "gemini")


def _serialize(
    request: ChatRequest, deps: TranslationDeps, target: p.GoogleTarget
) -> _SerializeResult:
    cache_gate = _cache_marker_gate(request)
    if cache_gate is not None:
        return Error(cache_gate)
    system_error = _system_gate(request, deps)
    if system_error is not None:
        return Error(system_error)
    match _structured_output_entries(request, deps):
        case Result(tag="ok", ok=(structured_entries, schema_prompt)):
            pass
        case Result(error=err):
            return Error(err)
    match _generation_config(request, deps, target, structured_entries):
        case Result(tag="ok", ok=generation_config):
            pass
        case Result(error=config_err):
            return Error(config_err)

    messages = request.messages
    if schema_prompt is not None:
        messages = with_schema_prompt(messages, schema_prompt)
    contents = _contents(messages, request, target)
    if isinstance(contents, TranslationError):
        return Error(contents)
    return _assemble_body(request, contents, generation_config)


def _assemble_body(
    request: ChatRequest,
    contents: list[PlainJson],
    generation_config: dict[str, PlainJson],
) -> _SerializeResult:
    body: Body = {"contents": contents}
    system_instruction = serialize_system(request.system)
    if system_instruction is not None:
        body = {**body, "system_instruction": system_instruction}
    tools = serialize_tools(request.tools)
    if isinstance(tools, TranslationError):
        return Error(tools)
    if tools:
        body = {**body, "tools": tools}
    match request.tool_choice:
        case Option(tag="some", some=choice):
            body = {**body, "toolConfig": tool_config(choice)}
        case _:
            pass
    if generation_config:
        body = {**body, "generationConfig": generation_config}
    return Ok(body)


def _contents(
    messages: Block[Message], request: ChatRequest, target: p.GoogleTarget
) -> list[PlainJson] | TranslationError:
    if len(messages) == 0 and len(request.system) > 0:
        # v1 _transform_system_message: blank "." user message so gemini
        # accepts a system-only request.
        return [{"role": "user", "parts": [{"text": "."}]}]
    return serialize_contents(messages, request.model, target)


def _system_gate(
    request: ChatRequest, deps: TranslationDeps
) -> TranslationError | None:
    if len(request.system) == 0:
        return None
    if deps.capability_flag(request.model, "supports_system_messages") is True:
        return None
    return TranslationError.of_unsupported(
        "system messages on a model without supports_system_messages; v1 folds them into user turns"
    )


def _cache_marker_gate(request: ChatRequest) -> TranslationError | None:
    has_marker = any(entry.cache.is_some() for entry in request.system) or any(
        _message_has_marker(message) for message in request.messages
    )
    if not has_marker:
        return None
    total = sum(len(entry.text.encode("utf-8")) for entry in request.system)
    total = total + _PER_MESSAGE_TOKEN_MARGIN * (
        len(request.system) + len(request.messages)
    )
    for message in request.messages:
        counted = _message_bytes(message)
        if isinstance(counted, TranslationError):
            return counted
        total = total + counted
    if total >= _CACHE_MARKER_TOKEN_LIMIT:
        return TranslationError.of_unsupported(
            "cache_control content whose conservative token bound reaches gemini's cache minimum; v1's check_and_create_cache performs network I/O"
        )
    return None


def _message_has_marker(message: Message) -> bool:
    return any(
        (block.tag == "text" and block.text.cache.is_some())
        or (block.tag == "tool_result" and block.tool_result.cache.is_some())
        or (block.tag == "thinking" and block.thinking.cache.is_some())
        or (block.tag == "tool_use" and block.tool_use.cache.is_some())
        or (block.tag == "image" and block.image.cache.is_some())
        for block in message.content
    )


def _message_bytes(message: Message) -> int | TranslationError:
    total = 0
    for block in message.content:
        counted = _block_bytes(block)
        if isinstance(counted, TranslationError):
            return counted
        total = total + counted
    return total


def _block_bytes(block: ContentBlock) -> int | TranslationError:
    match block.tag:
        case "text":
            return len(block.text.text.encode("utf-8"))
        case "thinking":
            return len(block.thinking.thinking.encode("utf-8"))
        case "redacted_thinking":
            return len(block.redacted_thinking.data.encode("utf-8"))
        case "tool_use":
            arguments = json.dumps(block.tool_use.arguments.value)
            return len(block.tool_use.name.encode("utf-8")) + len(
                arguments.encode("utf-8")
            )
        case "tool_result":
            content = block.tool_result.content
            if content.tag == "text":
                return len(content.text.encode("utf-8"))
            return sum(len(part.text.encode("utf-8")) for part in content.parts)
        case "image":
            # v1 charges DEFAULT_IMAGE_TOKEN_COUNT per image at zero text
            # bytes, and an unmarked image can sit inside the continuous
            # cached block; no byte bound exists, so media fails closed.
            return TranslationError.of_unsupported(
                "media beside cache_control markers; v1's check_and_create_cache token-counts images (and may create a cache)"
            )
    assert_never(block.tag)


_StructuredEntries = dict[str, PlainJson]
_StructuredResult = Result[tuple[_StructuredEntries, str | None], TranslationError]


def _structured_output_entries(
    request: ChatRequest, deps: TranslationDeps
) -> _StructuredResult:
    match request.response_format:
        case Option(tag="some", some=response_format):
            pass
        case _:
            return Ok(({}, None))
    match response_format.tag:
        case "text":
            return Ok(({"response_mime_type": "text/plain"}, None))
        case "json_object":
            return Ok(({"response_mime_type": "application/json"}, None))
        case "json_schema":
            return _json_schema_entries(
                response_format.json_schema.schema.value, request, deps
            )
    assert_never(response_format.tag)


def _json_schema_entries(
    raw_schema: PlainJson, request: ChatRequest, deps: TranslationDeps
) -> _StructuredResult:
    entries: _StructuredEntries = {"response_mime_type": "application/json"}
    if not isinstance(raw_schema, dict):
        # v1 only attaches the schema when it is a dict; the mime sticks.
        return Ok((entries, None))
    schema = strip_strict(copy.deepcopy(raw_schema))
    if p.supports_response_json_schema(request.model):
        # gemini 2.x+: standard JSON Schema, passed through (_build_json_schema
        # is the identity).
        return Ok(({**entries, "response_json_schema": schema}, None))
    cleaned = strip_additional_properties(schema)
    if not isinstance(cleaned, dict):
        return Ok((entries, None))
    built = build_vertex_schema(cleaned, add_property_ordering=True)
    if isinstance(built, TranslationError):
        return Error(built)
    if deps.supports_capability(request.model, "supports_response_schema"):
        return Ok(({**entries, "response_schema": built}, None))
    # v1 response_schema_prompt consults litellm.custom_prompt_dict; the seam
    # only routes here when that ambient dict is empty, so the default prompt
    # applies (str(dict) formatting AND the trailing spaces included).
    prompt = f"Use this JSON schema: \n    ```json \n    {built}\n    ```"
    return Ok((entries, prompt))


def _generation_config(
    request: ChatRequest,
    deps: TranslationDeps,
    target: p.GoogleTarget,
    structured_entries: _StructuredEntries,
) -> Result[_StructuredEntries, TranslationError]:
    sampling = p.sampling_entries(request, deps, target)
    if isinstance(sampling, TranslationError):
        return Error(sampling)
    entries: _StructuredEntries = {**sampling, **structured_entries}
    thinking = _thinking_entries(request)
    if isinstance(thinking, TranslationError):
        return Error(thinking)
    if thinking is not None:
        entries = {**entries, "thinkingConfig": thinking}
    return Ok(entries)


def _thinking_entries(
    request: ChatRequest,
) -> dict[str, PlainJson] | None | TranslationError:
    effort = request.reasoning_effort.default_value(None)
    has_thinking = request.thinking.is_some()
    if effort is not None and has_thinking and p.is_gemini_3_or_newer(request.model):
        return TranslationError.of_unsupported(
            "reasoning_effort + thinking on gemini-3 hits v1's thinking_level conflict check (raises)"
        )
    config: dict[str, PlainJson] | TranslationError | None = None
    if effort is not None:
        config = p.effort_thinking_config(effort, request.model)
        if isinstance(config, TranslationError):
            return config
    match request.thinking:
        case Option(tag="some", some=thinking_param):
            # request order rides the seam's fixed field order: thinking is
            # processed after reasoning_effort in v1's loop, so it wins.
            config = p.thinking_param_config(thinking_param, request.model)
            if isinstance(config, TranslationError):
                return config
        case _:
            pass
    return config
