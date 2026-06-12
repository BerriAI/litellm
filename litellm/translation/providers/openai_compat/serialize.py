"""Serialize the IR into an OpenAI ``/v1/chat/completions`` request body.

v1's ``OpenAIGPTConfig.map_openai_params`` + ``transform_request``
(llms/openai/chat/gpt_transformation.py:429-456) is a near-passthrough with
exactly five touches: string ``image_url`` -> object (+ litellm ``format``
strip), pdf-URL ``file`` inlining (sync I/O, fails closed before this module:
the inbound schema has no file part), ``cache_control`` stripped recursively
from messages and tools, ``max_retries`` popped (an SDK kwarg, never in the
IR), and body assembly ``{model, messages, **optional_params}``. The shapes
this serializer admits are the ones the raw guard proved round-trip
losslessly; o-series and gpt-5 models fail closed until their param families
(``OpenAIOSeriesConfig``/``OpenAIGPT5Config``) are ported.
"""

from __future__ import annotations

from collections.abc import Callable

from expression import Error, Ok, Option, Result
from expression.collections import Block
from typing_extensions import assert_never

from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH

from ...deps import TranslationDeps
from ...errors import TranslationError
from ...ir import Body, ChatRequest, PlainJson, ResponseFormat, ToolChoice, ToolDef
from . import params as p
from .messages import serialize_messages

_SerializeResult = Result[Body, TranslationError]

_GateFn = Callable[[ChatRequest, TranslationDeps], str | None]
_DeltasFn = Callable[[Body, ChatRequest], Body]
_Serializer = Callable[[ChatRequest, TranslationDeps], _SerializeResult]


def make_gated_serializer(gate: _GateFn, with_deltas: _DeltasFn) -> _Serializer:
    """The own-module ``serialize_request`` shape — params gate ->
    ``assemble_body`` -> provider deltas — as ONE factory instead of five
    identical wrappers (critic-wave2b-alpha NIT-1). Consumers: deepseek,
    openrouter, hosted_vllm, fireworks_ai, huggingface. snowflake binds the
    chain explicitly: its deltas return a Result (the fail-closed tool arm),
    so it is deliberately not a row here."""

    def serialize_request(
        request: ChatRequest, deps: TranslationDeps
    ) -> _SerializeResult:
        reason = gate(request, deps)
        if reason is not None:
            return Error(TranslationError.of_unsupported(reason))
        return assemble_body(request).map(lambda body: with_deltas(body, request))

    return serialize_request


def serialize_request(request: ChatRequest, deps: TranslationDeps) -> _SerializeResult:
    reason = (
        p.unsupported_model_family(request.model)
        or p.unsupported_params(request)
        or p.unsupported_response_format(request)
    )
    if reason is not None:
        return Error(TranslationError.of_unsupported(reason))
    return assemble_body(request)


def assemble_body(request: ChatRequest) -> _SerializeResult:
    """The gate-free five-touch body assembly, for consumers (xai) whose v1
    config inherits OpenAIGPTConfig's transform_request but replaces the
    openai param gates with their own."""
    messages = serialize_messages(request)
    if isinstance(messages, TranslationError):
        return Error(messages)
    body: Body = {
        "model": request.model,
        "messages": messages,
        **_present(
            stream=True if request.stream else None,
            **_max_tokens_fields(request),
            temperature=request.params.temperature.default_value(None),
            top_p=request.params.top_p.default_value(None),
            stop=list(request.params.stop) if len(request.params.stop) > 0 else None,
            tools=_tools_json(request.tools),
            tool_choice=_tool_choice_json(request.tool_choice),
            parallel_tool_calls=request.parallel_tool_calls.default_value(None),
            response_format=_response_format_json(request.response_format),
        ),
    }
    return Ok(body)


def _present(**fields: PlainJson | None) -> dict[str, PlainJson]:
    return {key: value for key, value in fields.items() if value is not None}


def _max_tokens_fields(request: ChatRequest) -> dict[str, PlainJson | None]:
    """Re-emit the caller's original key; the raw guard rejects requests
    carrying both, so ``max_completion_tokens`` being set means it was the
    one sent (``max_tokens`` then only holds the collapsed copy)."""
    match request.params.max_completion_tokens:
        case Option(tag="some", some=value):
            return {"max_completion_tokens": value}
        case _:
            return {"max_tokens": request.params.max_tokens.default_value(None)}


def _tools_json(tools: Block[ToolDef]) -> PlainJson | None:
    if len(tools) == 0:
        return None
    return [_tool_json(tool) for tool in tools]


def _tool_json(tool: ToolDef) -> PlainJson:
    function: dict[str, PlainJson] = {"name": tool.name}
    match tool.description:
        case Option(tag="some", some=description):
            function = {**function, "description": description}
        case _:
            pass
    match tool.parameters:
        case Option(tag="some", some=parameters):
            function = {**function, "parameters": _strip_cache(parameters.value, 0)}
        case _:
            pass
    match tool.strict:
        case Option(tag="some", some=strict):
            function = {**function, "strict": strict}
        case _:
            pass
    return {"type": "function", "function": function}


def _strip_cache(value: PlainJson, depth: int) -> PlainJson:
    """Mirror v1's ``filter_value_from_dict(tool, "cache_control")``: the key
    is removed at every nesting level, with the same recursion cap."""
    if depth > DEFAULT_MAX_RECURSE_DEPTH:
        return value
    if isinstance(value, dict):
        return {
            key: _strip_cache(item, depth + 1)
            for key, item in value.items()
            if key != "cache_control"
        }
    if isinstance(value, list):
        return [_strip_cache(item, depth + 1) for item in value]
    return value


def _tool_choice_json(choice_opt: Option[ToolChoice]) -> PlainJson | None:
    match choice_opt:
        case Option(tag="some", some=choice):
            pass
        case _:
            return None
    match choice.tag:
        case "auto":
            return "auto"
        case "required":
            return "required"
        case "none":
            return "none"
        case "specific":
            return {"type": "function", "function": {"name": choice.specific}}
    assert_never(choice.tag)


def _response_format_json(
    response_format_opt: Option[ResponseFormat],
) -> PlainJson | None:
    match response_format_opt:
        case Option(tag="some", some=response_format):
            pass
        case _:
            return None
    match response_format.tag:
        case "text":
            return {"type": "text"}
        case "json_object":
            return {"type": "json_object"}
        case "json_schema":
            spec = response_format.json_schema
            inner: dict[str, PlainJson] = {}
            match spec.name:
                case Option(tag="some", some=name):
                    inner = {**inner, "name": name}
                case _:
                    pass
            match spec.description:
                case Option(tag="some", some=description):
                    inner = {**inner, "description": description}
                case _:
                    pass
            inner = {**inner, "schema": spec.schema.value}
            match spec.strict:
                case Option(tag="some", some=strict):
                    inner = {**inner, "strict": strict}
                case _:
                    pass
            return {"type": "json_schema", "json_schema": inner}
    assert_never(response_format.tag)


def strip_function_strict(tool: PlainJson) -> PlainJson:
    """Drop the FUNCTION-LEVEL ``strict`` key from one tool (deeper
    ``strict`` keys untouched) — v1's ``function.pop("strict", None)``
    shape shared by xai (filter_value_from_dict at the function level) and
    fireworks_ai (_transform_tools). Lifted from xai/serialize.py when
    fireworks_ai became the second consumer."""
    if not isinstance(tool, dict):
        return tool
    function = tool.get("function")
    if not isinstance(function, dict):
        return tool
    return {
        **tool,
        "function": {key: value for key, value in function.items() if key != "strict"},
    }
