"""Serialize the IR onto the generateContent wire format.

ONE serializer family for both google routes; the ``target`` parameter is the
researcher-2 drift list and nothing else: AI Studio drops ``top_k`` (or fails
like v1 without drop_params), refuses https media (v1 downloads it), and
forwards function-call ids on gemini-3+; vertex passes https media through as
``file_data`` and would attach ``labels`` (not expressible in the IR). Auth,
hosts, and API versions are envelope and never appear here.

Fail-closed shapes (each names the v1 path): cache markers that could reach
``check_and_create_cache``'s network call, gemini-3 thinking budgets (read an
ambient litellm global), reasoning_effort xhigh/max (v1 raises), and anything
the message/tool converters decline.
"""

from __future__ import annotations

import copy

from expression import Error, Ok, Option, Result
from expression.collections import Block
from typing_extensions import assert_never

from ...deps import TranslationDeps
from ...errors import TranslationError
from ...ir import Body, ChatRequest, Message, PlainJson
from . import params as p
from .messages import serialize_contents, serialize_system, with_schema_prompt
from .schema import build_vertex_schema, strip_additional_properties, strip_strict
from .tools import serialize_tools, tool_config

_SerializeResult = Result[Body, TranslationError]

# Gemini's context-cache minimum is 1024 TOKENS; one token always spans at
# least one character, so < 1024 chars of cache-marked text guarantees v1
# skips the cache-create network call and ignores the markers.
_CACHE_MARKER_CHAR_LIMIT = 1024


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


def _system_gate(request: ChatRequest, deps: TranslationDeps) -> TranslationError | None:
    if len(request.system) == 0:
        return None
    if deps.capability_flag(request.model, "supports_system_messages") is True:
        return None
    return TranslationError.of_unsupported(
        "system messages on a model without supports_system_messages; v1 folds them into user turns"
    )


def _cache_marker_gate(request: ChatRequest) -> TranslationError | None:
    system_chars = sum(
        len(entry.text) for entry in request.system if entry.cache.is_some()
    )
    message_chars = 0
    for message in request.messages:
        counted = _marked_message_chars(message)
        if isinstance(counted, TranslationError):
            return counted
        message_chars = message_chars + counted
    if system_chars + message_chars >= _CACHE_MARKER_CHAR_LIMIT:
        return TranslationError.of_unsupported(
            "cache_control content at or above gemini's cache minimum; v1's check_and_create_cache performs network I/O"
        )
    return None


def _block_text_chars(message: Message) -> int:
    total = 0
    for block in message.content:
        if block.tag == "text":
            total = total + len(block.text.text)
        elif block.tag == "thinking":
            total = total + len(block.thinking.thinking)
        elif block.tag == "tool_result":
            content = block.tool_result.content
            if content.tag == "text":
                total = total + len(content.text)
            else:
                total = total + sum(len(part.text) for part in content.parts)
    return total


def _marked_message_chars(message: Message) -> int | TranslationError:
    has_marker = any(
        (block.tag == "text" and block.text.cache.is_some())
        or (block.tag == "tool_result" and block.tool_result.cache.is_some())
        or (block.tag == "thinking" and block.thinking.cache.is_some())
        for block in message.content
    )
    media_marked = any(
        block.tag == "image" and block.image.cache.is_some()
        for block in message.content
    )
    if media_marked:
        return TranslationError.of_unsupported(
            "cache_control on media blocks; v1's check_and_create_cache token-counts them (and may create a cache)"
        )
    if not has_marker:
        return 0
    return _block_text_chars(message)


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
    # applies (str(dict) formatting included).
    prompt = """Use this JSON schema:
    ```json
    {}
    ```""".format(built)
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
