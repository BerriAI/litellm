"""Serialize the IR into an Anthropic ``/v1/messages`` request body.

Pure given injected deps, and fallible by design: any shape v1 resolves
through an unported path (schema ``$ref`` inlining, body-order-dependent
parameter interplay, gated sampling params without drop_params) returns
``unsupported`` so the dispatch seam falls back to v1 instead of silently
dropping a feature. The emitted body matches v1's ``map_openai_params`` +
``transform_request`` output for the supported surface, including the
``json_mode`` marker v1 leaves at the transform seam (the engine pops it
before the wire, exactly like v1's HTTP handler).
"""

from __future__ import annotations

import copy
import json
from typing import Dict, List, Mapping, Optional, Tuple

from typing_extensions import assert_never

from expression import Error, Ok, Option, Result
from expression.collections import Block

from ...deps import TranslationDeps
from ...errors import TranslationError
from ...ir import (
    Body,
    ChatRequest,
    PlainJson,
    ResponseFormat,
    SystemText,
    ToolChoice,
    has_tool_blocks,
)
from . import params as p
from .messages import serialize_messages
from .tools import (
    cache_json,
    dummy_tool,
    filter_output_schema,
    request_name_maps,
    response_format_tool,
    serialize_tool,
)

_SerializeResult = Result[Body, TranslationError]


def serialize_request(request: ChatRequest, deps: TranslationDeps) -> _SerializeResult:
    if len(request.messages) == 0:
        return Error(TranslationError.of_unsupported("no non-system messages: v1 raises unless modify_params"))
    match p.map_thinking(request, deps):
        case Result(tag="ok", ok=(thinking_json, output_config)):
            pass
        case Result(error=err):
            return Error(err)
    match _response_format_fields(request, deps):
        case Result(tag="ok", ok=(rf_tool, rf_tool_choice, output_format, json_mode)):
            pass
        case Result(error=rf_err):
            return Error(rf_err)
    match _sampling_fields(request, deps):
        case Result(tag="ok", ok=sampling):
            pass
        case Result(error=sampling_err):
            return Error(sampling_err)

    name_forward, _ = request_name_maps(request.tools)
    tools = _tools_json(request, rf_tool, name_forward)
    tool_choice = rf_tool_choice if rf_tool_choice is not None else _tool_choice_json(request, name_forward)
    body: Body = {
        "model": request.model,
        "messages": serialize_messages(request.messages, name_forward),
        "max_tokens": _max_tokens(request, thinking_json, deps),
        **sampling,
        **_present(
            stop_sequences=p.filter_stop(request.params.stop, deps),
            stream=True if request.stream else None,
            system=_system_json(request.system),
            tools=tools,
            tool_choice=tool_choice,
            metadata=_metadata(request),
            thinking=thinking_json,
            output_config=output_config,
            output_format=output_format,
            json_mode=json_mode,
        ),
    }
    return Ok(body)


def _present(**fields: Optional[PlainJson]) -> Dict[str, PlainJson]:
    return {key: value for key, value in fields.items() if value is not None}


def _sampling_fields(request: ChatRequest, deps: TranslationDeps) -> Result[Dict[str, PlainJson], TranslationError]:
    gathered: Dict[str, PlainJson] = {}
    for param, value_opt in (
        ("temperature", request.params.temperature),
        ("top_p", request.params.top_p),
        ("top_k", request.params.top_k),
    ):
        match value_opt:
            case Option(tag="some", some=value):
                pass
            case _:
                continue
        match p.gate_sampling_param(request.model, param, value, deps):
            case Result(tag="ok", ok=Option(tag="some", some=kept)):
                gathered = {**gathered, param: kept}
            case Result(tag="ok", ok=_):
                continue
            case Result(error=err):
                return Error(err)
    return Ok(gathered)


def _max_tokens(request: ChatRequest, thinking_json: Optional[PlainJson], deps: TranslationDeps) -> int:
    explicit = p.bump_max_tokens_for_thinking(request.params.max_tokens, thinking_json)
    match explicit:
        case Option(tag="some", some=value):
            return value
        case _:
            return p.default_max_tokens(request.model, deps)


def _system_json(system: Block[SystemText]) -> Optional[PlainJson]:
    if len(system) == 0:
        return None
    blocks: List[PlainJson] = []
    for text in system:
        base: Dict[str, PlainJson] = {"type": "text", "text": text.text}
        match text.cache:
            case Option(tag="some", some=cache):
                blocks.append(  # nosemgrep: translation-no-mutation
                    {**base, "cache_control": cache_json(cache)}
                )
            case _:
                blocks.append(base)  # nosemgrep: translation-no-mutation
    return blocks


def _tools_json(
    request: ChatRequest,
    rf_tool: Optional[PlainJson],
    name_forward: Mapping[str, str],
) -> Optional[PlainJson]:
    user_tools = [serialize_tool(tool) for tool in request.tools]
    rf_tools = [rf_tool] if rf_tool is not None else []
    tools: List[PlainJson] = [*user_tools, *rf_tools]
    if tools:
        return _apply_forward_map(tools, name_forward)
    if has_tool_blocks(request.messages):
        return [dummy_tool()]
    return None


def _apply_forward_map(tools: List[PlainJson], forward: Mapping[str, str]) -> List[PlainJson]:
    if not forward:
        return tools
    return [
        {**tool, "name": forward[tool["name"]]}
        if isinstance(tool, dict)
        and tool.get("type") == "custom"
        and isinstance(tool.get("name"), str)
        and tool["name"] in forward
        else tool
        for tool in tools
    ]


def _tool_choice_json(request: ChatRequest, name_forward: Mapping[str, str]) -> Optional[PlainJson]:
    """v1 ``_map_tool_choice`` plus the request-level name forward map."""
    parallel: Optional[bool] = request.parallel_tool_calls.default_value(None)
    choice_json: Optional[Dict[str, PlainJson]] = None
    is_none_choice = False
    match request.tool_choice:
        case Option(tag="some", some=choice):
            choice_json, is_none_choice = _choice_base(choice, name_forward)
        case _:
            choice_json = None
    if parallel is None:
        return choice_json
    if is_none_choice:
        return choice_json
    if choice_json is not None:
        return {**choice_json, "disable_parallel_tool_use": not parallel}
    return {"type": "auto", "disable_parallel_tool_use": not parallel}


def _choice_base(choice: ToolChoice, name_forward: Mapping[str, str]) -> Tuple[Dict[str, PlainJson], bool]:
    match choice.tag:
        case "auto":
            return {"type": "auto"}, False
        case "required":
            return {"type": "any"}, False
        case "none":
            return {"type": "none"}, True
        case "specific":
            name = name_forward.get(choice.specific, choice.specific)
            return {"type": "tool", "name": name}, False
        case never:
            assert_never(never)


def _metadata(request: ChatRequest) -> Optional[PlainJson]:
    match request.user:
        case Option(tag="some", some=user):
            if p.valid_user_id(user):
                return {"user_id": user}
            return None  # v1 silently skips invalid user ids
        case _:
            return None


_RfFields = Tuple[Optional[PlainJson], Optional[PlainJson], Optional[PlainJson], Optional[bool]]
"""(json_tool_call tool, forced tool_choice, output_format, json_mode)."""


def _response_format_fields(request: ChatRequest, deps: TranslationDeps) -> Result[_RfFields, TranslationError]:
    match request.response_format:
        case Option(tag="some", some=response_format):
            pass
        case _:
            return Ok((None, None, None, None))
    if p.uses_output_format(request.model):
        return _output_format_fields(response_format)
    return _json_tool_fields(request, response_format)


def _output_format_fields(
    response_format: ResponseFormat,
) -> Result[_RfFields, TranslationError]:
    if response_format.tag != "json_schema":
        # json_object / text on output_format models: no schema, but v1 still
        # marks json_mode.
        return Ok((None, None, None, True))
    schema = response_format.json_schema.schema.value
    if '"$ref"' in json.dumps(schema):
        return Error(TranslationError.of_unsupported("response_format schema with $ref needs v1's def inlining"))
    # v1 pops top-level $defs/definitions before filtering (they only exist to
    # back $refs, which we just proved absent).
    if isinstance(schema, dict):
        schema = {key: value for key, value in schema.items() if key not in ("$defs", "definitions")}
    filtered = filter_output_schema(copy.deepcopy(schema))
    return Ok((None, None, {"type": "json_schema", "schema": filtered}, True))


def _json_tool_fields(request: ChatRequest, response_format: ResponseFormat) -> Result[_RfFields, TranslationError]:
    if response_format.tag != "json_schema":
        # On the json-tool models v1 ignores type "text" AND type "json_object"
        # (no schema extracted -> no tool, no json_mode -> full no-op).
        return Ok((None, None, None, None))
    conflicted = len(request.tools) > 0 or request.tool_choice.is_some() or request.parallel_tool_calls.is_some()
    if conflicted:
        return Error(
            TranslationError.of_unsupported(
                "response_format json tool path combined with tools/tool_choice is body-order-dependent in v1"
            )
        )
    schema = copy.deepcopy(response_format.json_schema.schema.value)
    tool = response_format_tool(schema)
    forced_choice: Optional[PlainJson] = None
    if not p.thinking_signaled(request):
        forced_choice = {"name": "json_tool_call", "type": "tool"}
    return Ok((tool, forced_choice, None, True))
