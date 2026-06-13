"""Parse an OpenAI Responses request body into the chat IR.

``boundary.parse`` validates the untyped body against the frozen wire models;
this module converts the validated models into IR values and applies the
residual fail-closed checks (every field v1 serves through an unported path
becomes an ``unsupported`` naming that path, never a silent drop). Top-level
``null`` fields are stripped first, matching the seam's treatment of an
explicit ``null`` as an absent parameter.

Forward map (researcher-6 §2.1): ``input`` -> messages (per-item dispatch in
``input_items``), ``instructions`` -> system, function ``tools`` -> ToolDef,
``tool_choice`` -> ToolChoice (incl. the Cursor ``{type:tool/any}`` -> required
normalization), ``reasoning.effort`` -> ReasoningEffort, ``max_output_tokens``
-> max_tokens, ``text.format`` -> response_format, temperature/top_p/
parallel_tool_calls/user verbatim. The wide must-fall-back surface
(previous_response_id, store/background, hosted tools, MCP, reasoning summary,
metadata, include, service_tier) is named in ``_UNPORTED_FIELDS`` and the
per-item / per-tool checks below.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

from expression import Error, Nothing, Ok, Option, Result, Some
from expression.collections import Block

from ... import boundary
from ...errors import ParseResult, TranslationError
from ...ir import (
    ChatRequest,
    InferenceParams,
    JsonBlob,
    JsonSchemaSpec,
    Message,
    ReasoningEffort,
    ResponseFormat,
    SystemText,
    ToolChoice,
    ToolDef,
)
from .input_items import convert_input_items, convert_string_input
from .schema import (
    FunctionToolIn,
    ReasoningIn,
    ResponsesRequestIn,
    TextFormatIn,
    TextIn,
    ToolChoiceNamedIn,
)

_T = TypeVar("_T")

_UNPORTED_FIELDS: tuple[tuple[str, str], ...] = (
    ("previous_response_id", "responses previous_response_id; needs session state"),
    ("metadata", "responses metadata; no chat-IR home"),
    ("service_tier", "responses service_tier passthrough"),
    ("include", "responses include[]"),
    ("store", "responses store (server-side persistence)"),
    ("background", "responses background (polling)"),
    ("truncation", "responses truncation"),
)


def parse_request(raw: Mapping[str, object]) -> ParseResult:
    present = {key: value for key, value in raw.items() if value is not None}
    match boundary.parse(ResponsesRequestIn, present):
        case Result(tag="ok", ok=wire):
            return _to_ir(wire)
        case Result(error=err):
            return Error(err)


def _to_ir(wire: ResponsesRequestIn) -> ParseResult:
    unported = _unported_error(wire)
    if unported is not None:
        return Error(unported)
    match _parse_messages(wire):
        case Result(tag="ok", ok=messages):
            pass
        case Result(error=messages_err):
            return Error(messages_err)
    match _parse_tools(wire.tools):
        case Result(tag="ok", ok=tools):
            pass
        case Result(error=tools_err):
            return Error(tools_err)
    match _parse_response_format(wire.text):
        case Result(tag="ok", ok=response_format):
            pass
        case Result(error=text_err):
            return Error(text_err)
    match _parse_reasoning(wire.reasoning):
        case Result(tag="ok", ok=reasoning_effort):
            pass
        case Result(error=reasoning_err):
            return Error(reasoning_err)
    return Ok(
        ChatRequest(
            model=wire.model,
            system=_system(wire.instructions),
            messages=messages,
            tools=tools,
            tool_choice=_parse_tool_choice(wire.tool_choice),
            parallel_tool_calls=_option(wire.parallel_tool_calls),
            response_format=response_format,
            thinking=Nothing,
            reasoning_effort=reasoning_effort,
            user=_option(wire.user),
            params=_parse_params(wire),
            stream=wire.stream is True,
        )
    )


def _unported_error(wire: ResponsesRequestIn) -> TranslationError | None:
    for field, reason in _UNPORTED_FIELDS:
        if getattr(wire, field) is not None:
            return TranslationError.of_unsupported(f"{reason}; v1 handles it")
    return None


def _option(value: _T | None) -> Option[_T]:
    return Some(value) if value is not None else Nothing


def _system(instructions: str | None) -> Block[SystemText]:
    if instructions is None:
        return Block.empty()
    return Block.of_seq([SystemText(text=instructions, cache=Nothing)])


def _parse_messages(
    wire: ResponsesRequestIn,
) -> Result[Block[Message], TranslationError]:
    if isinstance(wire.input, str):
        return Ok(convert_string_input(wire.input))
    return convert_input_items(wire.input)


def _parse_tools(
    tools: list[FunctionToolIn | object] | None,
) -> Result[Block[ToolDef], TranslationError]:
    if tools is None:
        return Ok(Block.empty())
    defs: list[ToolDef] = []
    for tool in tools:
        if not isinstance(tool, FunctionToolIn):
            return Error(
                TranslationError.of_unsupported(
                    "responses hosted tools (web_search/file_search/computer_use/"
                    "code_interpreter/image_generation/mcp); v1 handles them"
                )
            )
        match _parse_tool(tool):
            case Result(tag="ok", ok=tool_def):
                defs.append(tool_def)  # nosemgrep: translation-no-mutation
            case Result(error=err):
                return Error(err)
    return Ok(Block.of_seq(defs))


def _parse_tool(tool: FunctionToolIn) -> Result[ToolDef, TranslationError]:
    extras = (tool.defer_loading, tool.allowed_callers, tool.input_examples)
    if any(extra is not None for extra in extras):
        return Error(
            TranslationError.of_unsupported(
                "tool defer_loading/allowed_callers/input_examples; v1 handles them"
            )
        )
    if tool.cache_control is not None:
        return Error(
            TranslationError.of_unsupported(
                "responses tool cache_control; v1 handles it"
            )
        )
    match _parse_parameters(tool.parameters):
        case Result(tag="ok", ok=parameters):
            return Ok(
                ToolDef(
                    name=tool.name,
                    description=_option(tool.description),
                    parameters=parameters,
                    cache=Nothing,
                    strict=_option(tool.strict),
                )
            )
        case Result(error=err):
            return Error(err)


def _parse_parameters(
    parameters: object | None,
) -> Result[Option[JsonBlob], TranslationError]:
    if parameters is None:
        return Ok(Nothing)
    match boundary.as_plain_json(parameters):
        case Result(tag="ok", ok=copied) if isinstance(copied, dict):
            return Ok(Some(JsonBlob(value=copied)))
        case Result(tag="ok"):
            return Error(
                TranslationError.of_unsupported(
                    "non-object tool parameters; v1 handles it"
                )
            )
        case Result(error=reason):
            return Error(TranslationError.of_unsupported(f"tool parameters: {reason}"))


def _parse_tool_choice(choice: str | ToolChoiceNamedIn | None) -> Option[ToolChoice]:
    if choice is None:
        return Nothing
    if isinstance(choice, str):
        return _named_tool_choice(choice, None)
    return _named_tool_choice(choice.type, choice.name)


def _named_tool_choice(kind: str, name: str | None) -> Option[ToolChoice]:
    if kind == "auto":
        return Some(ToolChoice.of_auto())
    if kind == "none":
        return Some(ToolChoice.of_none())
    if kind in ("required", "tool", "any"):
        return Some(ToolChoice.of_required())
    if kind == "function" and name is not None:
        return Some(ToolChoice.of_specific(name))
    if kind == "function":
        return Some(ToolChoice.of_required())
    return Nothing


def _parse_response_format(
    text: TextIn | None,
) -> Result[Option[ResponseFormat], TranslationError]:
    if text is None or text.format is None:
        return Ok(Nothing)
    return _format(text.format)


def _format(fmt: TextFormatIn) -> Result[Option[ResponseFormat], TranslationError]:
    if fmt.type == "text":
        return Ok(Some(ResponseFormat.of_text()))
    if fmt.type == "json_object":
        return Ok(Some(ResponseFormat.of_json_object()))
    return _json_schema(fmt)


def _json_schema(fmt: TextFormatIn) -> Result[Option[ResponseFormat], TranslationError]:
    match boundary.as_plain_json(fmt.schema_ if fmt.schema_ is not None else {}):
        case Result(tag="ok", ok=copied) if isinstance(copied, dict):
            spec = JsonSchemaSpec(
                schema=JsonBlob(value=copied),
                name=Some(fmt.name if fmt.name is not None else "response_schema"),
                description=Nothing,
                strict=Some(fmt.strict if fmt.strict is not None else False),
            )
            return Ok(Some(ResponseFormat.of_json_schema(spec)))
        case Result(tag="ok"):
            return Error(
                TranslationError.of_unsupported(
                    "responses text.format json_schema with non-object schema; v1 handles it"
                )
            )
        case Result(error=reason):
            return Error(
                TranslationError.of_unsupported(f"text.format schema: {reason}")
            )


def _parse_reasoning(
    reasoning: ReasoningIn | None,
) -> Result[Option[ReasoningEffort], TranslationError]:
    if reasoning is None:
        return Ok(Nothing)
    if reasoning.summary is not None or reasoning.generate_summary is not None:
        return Error(
            TranslationError.of_unsupported(
                "responses reasoning.summary; v1 handles it"
            )
        )
    if reasoning.effort is None:
        return Ok(Nothing)
    return Ok(Some(reasoning.effort))


def _parse_params(wire: ResponsesRequestIn) -> InferenceParams:
    return InferenceParams(
        max_tokens=_option(wire.max_output_tokens),
        temperature=_option(wire.temperature),
        top_p=_option(wire.top_p),
        top_k=Nothing,
        stop=Block.empty(),
        max_completion_tokens=Nothing,
    )
