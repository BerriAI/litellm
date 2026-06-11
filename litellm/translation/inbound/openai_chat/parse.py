"""Parse an OpenAI chat-completions request body into the IR.

``boundary.parse`` validates the untyped body against the frozen wire models
(accumulating every field failure, arktype-style); this module then converts
the validated models into IR values, with the residual semantic checks that
need more than shape. Top-level ``null`` fields are stripped first because the
v1 seam treats an explicit ``null`` exactly like an absent parameter.
"""

from __future__ import annotations

from typing import List, Mapping, Optional, TypeVar, Union

from expression import Error, Nothing, Ok, Option, Result, Some
from expression.collections import Block

from ... import boundary
from ...errors import ParseResult, TranslationError
from ...ir import (
    ChatRequest,
    InferenceParams,
    JsonBlob,
    JsonSchemaSpec,
    ReasoningEffort,
    ResponseFormat,
    ThinkingParam,
    ToolChoice,
    ToolDef,
)
from .messages import cache_of, convert_messages
from .schema import (
    ChatRequestIn,
    ReasoningEffortObjectIn,
    ResponseFormatIn,
    ThinkingIn,
    ToolChoiceNamedIn,
    ToolChoiceTypeOnlyIn,
    ToolIn,
)

_T = TypeVar("_T")


def parse_request(raw: Mapping[str, object]) -> ParseResult:
    present = {key: value for key, value in raw.items() if value is not None}
    match boundary.parse(ChatRequestIn, present):
        case Result(tag="ok", ok=wire):
            return _to_ir(wire)
        case Result(error=err):
            return Error(err)


def _to_ir(wire: ChatRequestIn) -> ParseResult:
    if (
        isinstance(wire.tool_choice, ToolChoiceTypeOnlyIn)
        and wire.tool_choice.type == "none"
        and wire.parallel_tool_calls is not None
    ):
        # v1 treats string-"none" and {"type": "none"} differently when
        # parallel_tool_calls rides along; only the string form is ported.
        return Error(
            TranslationError.of_unsupported("tool_choice {'type': 'none'} with parallel_tool_calls; v1 handles it")
        )
    match convert_messages(wire.messages):
        case Result(tag="ok", ok=(systems, messages)):
            pass
        case Result(error=err):
            return Error(err)
    match _parse_tools(wire.tools):
        case Result(tag="ok", ok=tools):
            pass
        case Result(error=tools_err):
            return Error(tools_err)
    match _parse_response_format(wire.response_format):
        case Result(tag="ok", ok=response_format):
            pass
        case Result(error=rf_err):
            return Error(rf_err)
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
            parallel_tool_calls=_option(wire.parallel_tool_calls),
            response_format=response_format,
            thinking=thinking,
            reasoning_effort=_parse_reasoning_effort(wire.reasoning_effort),
            user=_option(wire.user),
            params=_parse_params(wire),
            stream=wire.stream is True,
        )
    )


def _option(value: Optional[_T]) -> Option[_T]:
    return Some(value) if value is not None else Nothing


def _parse_tools(
    tools: Optional[List[ToolIn]],
) -> Result[Block[ToolDef], TranslationError]:
    if tools is None:
        return Ok(Block.empty())
    defs: List[ToolDef] = []
    for tool in tools:
        match _parse_tool(tool):
            case Result(tag="ok", ok=tool_def):
                defs.append(tool_def)  # nosemgrep: translation-no-mutation
            case Result(error=err):
                return Error(err)
    return Ok(Block.of_seq(defs))


def _parse_tool(tool: ToolIn) -> Result[ToolDef, TranslationError]:
    extras = (
        tool.defer_loading,
        tool.allowed_callers,
        tool.input_examples,
        tool.function.defer_loading,
        tool.function.allowed_callers,
        tool.function.input_examples,
    )
    if any(extra is not None for extra in extras):
        return Error(
            TranslationError.of_unsupported("tool defer_loading/allowed_callers/input_examples; v1 handles them")
        )
    cache = cache_of(tool.cache_control)
    if cache.is_none():
        cache = cache_of(tool.function.cache_control)
    match _parse_tool_parameters(tool.function.parameters):
        case Result(tag="ok", ok=parameters):
            return Ok(
                ToolDef(
                    name=tool.function.name,
                    description=_option(tool.function.description),
                    parameters=parameters,
                    cache=cache,
                )
            )
        case Result(error=err):
            return Error(err)


def _parse_tool_parameters(
    parameters: Optional[object],
) -> Result[Option[JsonBlob], TranslationError]:
    if parameters is None:
        return Ok(Nothing)
    if not isinstance(parameters, dict):
        return Error(TranslationError.of_unsupported("non-object tool parameters; v1 handles them"))
    if "definitions" in parameters or "components" in parameters:
        return Error(TranslationError.of_unsupported("legacy $defs (definitions/components) need v1's schema inlining"))
    match boundary.as_plain_json(parameters):
        case Result(tag="ok", ok=copied):
            return Ok(Some(JsonBlob(value=copied)))
        case Result(error=reason):
            return Error(TranslationError.of_unsupported(f"tool parameters: {reason}"))


def _parse_tool_choice(
    choice: Union[str, ToolChoiceNamedIn, ToolChoiceTypeOnlyIn, None],
) -> Option[ToolChoice]:
    match choice:
        case None:
            return Nothing
        case "auto" | ToolChoiceTypeOnlyIn(type="auto"):
            return Some(ToolChoice.of_auto())
        case "required" | ToolChoiceTypeOnlyIn(type="required" | "any"):
            return Some(ToolChoice.of_required())
        case "none" | ToolChoiceTypeOnlyIn(type="none"):
            return Some(ToolChoice.of_none())
        case ToolChoiceNamedIn() as named:
            return Some(ToolChoice.of_specific(named.function.name))
        case _:
            return Nothing


def _parse_response_format(
    value: Optional[ResponseFormatIn],
) -> Result[Option[ResponseFormat], TranslationError]:
    if value is None:
        return Ok(Nothing)
    if value.type == "text":
        return Ok(Some(ResponseFormat.of_text()))
    if value.type == "json_object":
        return Ok(Some(ResponseFormat.of_json_object()))
    if value.json_schema is None:
        return Error(TranslationError.of_unsupported("response_format json_schema without a schema; v1 handles it"))
    match boundary.as_plain_json(value.json_schema.json_schema):
        case Result(tag="ok", ok=copied):
            if not isinstance(copied, dict):
                return Error(TranslationError.of_unsupported("non-object response_format schema; v1 handles it"))
            return Ok(Some(ResponseFormat.of_json_schema(JsonSchemaSpec(schema=JsonBlob(value=copied)))))
        case Result(error=reason):
            return Error(TranslationError.of_unsupported(f"response_format schema: {reason}"))


def _parse_thinking(
    value: Optional[ThinkingIn],
) -> Result[Option[ThinkingParam], TranslationError]:
    if value is None:
        return Ok(Nothing)
    if value.type == "enabled":
        return Ok(Some(ThinkingParam.of_enabled(_option(value.budget_tokens))))
    if value.budget_tokens is not None:
        return Error(
            TranslationError.of_unsupported(f"thinking type={value.type!r} with budget_tokens; v1 forwards it")
        )
    if value.type == "disabled":
        return Ok(Some(ThinkingParam.of_disabled()))
    return Ok(Some(ThinkingParam.of_adaptive()))


def _parse_reasoning_effort(
    value: Union[str, ReasoningEffortObjectIn, None],
) -> Option[ReasoningEffort]:
    if value is None:
        return Nothing
    if isinstance(value, ReasoningEffortObjectIn):
        return Some(value.effort)
    return Some(value)


def _parse_params(wire: ChatRequestIn) -> InferenceParams:
    max_tokens = wire.max_tokens if wire.max_tokens is not None else wire.max_completion_tokens
    return InferenceParams(
        max_tokens=_option(max_tokens),
        temperature=_option(wire.temperature),
        top_p=_option(wire.top_p),
        top_k=_option(wire.top_k),
        stop=_parse_stop(wire.stop),
    )


def _parse_stop(value: Union[str, List[str], None]) -> Block[str]:
    if value is None:
        return Block.empty()
    if isinstance(value, str):
        return Block.of_seq([value])
    return Block.of_seq(value)
