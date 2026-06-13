"""Parse an Anthropic Messages request body into the chat IR.

``boundary.parse`` validates the untyped body against the frozen wire models;
this module converts the validated models into IR values and applies the
residual fail-closed checks (every field v1 serves through an unported path
becomes an ``unsupported`` naming that path, never a silent drop). Top-level
``null`` fields are stripped first, matching the seam's treatment of an
explicit ``null`` as an absent parameter.

The IR is Anthropic-shaped, so the mapping is largely 1:1: ``stop_sequences``
renames to ``stop``, ``metadata.user_id`` to ``user``, ``thinking`` forwards
verbatim, ``tool_choice`` maps onto the four IR cases, and ``max_tokens`` is
required at the wire boundary (Anthropic rejects a request without it).
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
    ThinkingParam,
    ToolChoice,
    ToolDef,
)
from .messages import cache_of, convert_messages, convert_system
from .schema import (
    AnthropicMessagesRequestIn,
    ThinkingIn,
    ToolChoiceIn,
    ToolIn,
)

_T = TypeVar("_T")

_UNPORTED_FIELDS: tuple[tuple[str, str], ...] = (
    ("output_format", "anthropic structured outputs (output_format)"),
    ("output_config", "anthropic output_config (effort/format)"),
    ("mcp_servers", "anthropic mcp_servers"),
    ("container", "anthropic code-execution container"),
    ("context_management", "anthropic context-management polyfill"),
    ("reasoning_effort", "anthropic reasoning_effort passthrough"),
    ("inference_geo", "anthropic inference_geo passthrough"),
    ("speed", "anthropic speed (fast mode) passthrough"),
    ("cache_control", "anthropic top-level automatic cache_control"),
    ("service_tier", "anthropic service_tier passthrough"),
)


def parse_request(raw: Mapping[str, object]) -> ParseResult:
    present = {key: value for key, value in raw.items() if value is not None}
    match boundary.parse(AnthropicMessagesRequestIn, present):
        case Result(tag="ok", ok=wire):
            return _to_ir(wire)
        case Result(error=err):
            return Error(err)


def _to_ir(wire: AnthropicMessagesRequestIn) -> ParseResult:
    unported = _unported_error(wire)
    if unported is not None:
        return Error(unported)
    match convert_system(wire.system):
        case Result(tag="ok", ok=systems):
            pass
        case Result(error=system_err):
            return Error(system_err)
    match convert_messages(wire.messages):
        case Result(tag="ok", ok=messages):
            pass
        case Result(error=messages_err):
            return Error(messages_err)
    match _parse_tools(wire.tools):
        case Result(tag="ok", ok=tools):
            pass
        case Result(error=tools_err):
            return Error(tools_err)
    match _parse_thinking(wire.thinking):
        case Result(tag="ok", ok=thinking):
            pass
        case Result(error=thinking_err):
            return Error(thinking_err)
    return Ok(
        ChatRequest(
            model=wire.model,
            system=systems,
            messages=messages,
            tools=tools,
            tool_choice=_parse_tool_choice(wire.tool_choice),
            parallel_tool_calls=_parallel_tool_calls(wire.tool_choice),
            response_format=Nothing,
            thinking=thinking,
            reasoning_effort=Nothing,
            user=_user(wire),
            params=_parse_params(wire),
            stream=wire.stream is True,
        )
    )


def _unported_error(wire: AnthropicMessagesRequestIn) -> TranslationError | None:
    for field, reason in _UNPORTED_FIELDS:
        if getattr(wire, field) is not None:
            return TranslationError.of_unsupported(f"{reason}; v1 handles it")
    return None


def _option(value: _T | None) -> Option[_T]:
    return Some(value) if value is not None else Nothing


def _user(wire: AnthropicMessagesRequestIn) -> Option[str]:
    if wire.metadata is None:
        return Nothing
    return _option(wire.metadata.user_id)


def _parse_tools(
    tools: list[ToolIn] | None,
) -> Result[Block[ToolDef], TranslationError]:
    if tools is None:
        return Ok(Block.empty())
    defs: list[ToolDef] = []
    for tool in tools:
        match _parse_tool(tool):
            case Result(tag="ok", ok=tool_def):
                defs.append(tool_def)  # nosemgrep: translation-no-mutation
            case Result(error=err):
                return Error(err)
    return Ok(Block.of_seq(defs))


def _parse_tool(tool: ToolIn) -> Result[ToolDef, TranslationError]:
    extras = (tool.defer_loading, tool.allowed_callers, tool.input_examples)
    if any(extra is not None for extra in extras):
        return Error(
            TranslationError.of_unsupported(
                "tool defer_loading/allowed_callers/input_examples; v1 handles them"
            )
        )
    match _parse_input_schema(tool.input_schema):
        case Result(tag="ok", ok=parameters):
            return Ok(
                ToolDef(
                    name=tool.name,
                    description=_option(tool.description),
                    parameters=parameters,
                    cache=cache_of(tool.cache_control),
                    strict=Nothing,
                )
            )
        case Result(error=err):
            return Error(err)


def _parse_input_schema(
    schema: object | None,
) -> Result[Option[JsonBlob], TranslationError]:
    if schema is None:
        return Ok(Nothing)
    match boundary.as_plain_json(schema):
        case Result(tag="ok", ok=copied) if isinstance(copied, dict):
            return Ok(Some(JsonBlob(value=copied)))
        case Result(tag="ok"):
            return Error(
                TranslationError.of_unsupported(
                    "non-object tool input_schema; v1 handles it"
                )
            )
        case Result(error=reason):
            return Error(
                TranslationError.of_unsupported(f"tool input_schema: {reason}")
            )


def _parse_tool_choice(choice: ToolChoiceIn | None) -> Option[ToolChoice]:
    if choice is None:
        return Nothing
    if choice.type == "auto":
        return Some(ToolChoice.of_auto())
    if choice.type == "any":
        return Some(ToolChoice.of_required())
    if choice.type == "none":
        return Some(ToolChoice.of_none())
    if choice.name is None:
        return Nothing
    return Some(ToolChoice.of_specific(choice.name))


def _parallel_tool_calls(choice: ToolChoiceIn | None) -> Option[bool]:
    if choice is None or choice.disable_parallel_tool_use is None:
        return Nothing
    return Some(not choice.disable_parallel_tool_use)


def _parse_thinking(
    value: ThinkingIn | None,
) -> Result[Option[ThinkingParam], TranslationError]:
    if value is None:
        return Ok(Nothing)
    if value.summary is not None:
        return Error(
            TranslationError.of_unsupported(
                "thinking.summary (auto-summary); v1 handles it"
            )
        )
    if value.type == "enabled":
        return Ok(Some(ThinkingParam.of_enabled(_option(value.budget_tokens))))
    if value.type == "disabled":
        return Ok(Some(ThinkingParam.of_disabled()))
    return Ok(Some(ThinkingParam.of_adaptive()))


def _parse_params(wire: AnthropicMessagesRequestIn) -> InferenceParams:
    return InferenceParams(
        max_tokens=Some(wire.max_tokens),
        temperature=_option(wire.temperature),
        top_p=_option(wire.top_p),
        top_k=_option(wire.top_k),
        stop=_parse_stop(wire.stop_sequences),
        max_completion_tokens=Nothing,
    )


def _parse_stop(value: list[str] | None) -> Block[str]:
    if value is None:
        return Block.empty()
    return Block.of_seq(value)
