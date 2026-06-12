"""Serialize the IR into a Bedrock Converse request body.

Pure given injected deps, fallible by design: anything v1 resolves through an
unported path (non-Claude bedrock models, native structured outputs,
adaptive-effort output_config, fake_stream, dummy-tool injection under
modify_params) returns ``unsupported`` so the dispatch seam falls back to v1.
The emitted body matches v1's ``map_openai_params`` + ``transform_request``
output at the transform seam, including the ``additionalModelRequestFields.
stream`` marker that the seam's invocation of ``get_optional_params`` leaves
there (production pops ``stream`` before transform; the engine's wire step
mirrors that, exactly like the anthropic ``json_mode`` marker).
"""

from __future__ import annotations

from expression import Error, Ok, Option, Result
from expression.collections import Block
from typing_extensions import assert_never

from ...deps import TranslationDeps
from ...errors import TranslationError
from ...ir import (
    Body,
    ChatRequest,
    PlainJson,
    SystemText,
    ToolChoice,
    ToolDef,
    has_tool_blocks,
)
from ..anthropic import params as anthropic_params
from . import params as p
from .messages import serialize_messages
from .tools import (
    make_valid_bedrock_name,
    response_format_tool,
    serialize_tools,
    system_cache_point,
)

_SerializeResult = Result[Body, TranslationError]


def serialize_request(request: ChatRequest, deps: TranslationDeps) -> _SerializeResult:
    gate = _gate(request)
    if gate is not None:
        return Error(gate)
    match p.map_thinking(request, deps):
        case Result(tag="ok", ok=thinking_json):
            pass
        case Result(error=err):
            return Error(err)
    match _response_format_tool(request, deps):
        case Result(tag="ok", ok=(rf_tool, rf_choice)):
            pass
        case Result(error=rf_err):
            return Error(rf_err)
    tools = (
        Block.of_seq([*request.tools, rf_tool])
        if rf_tool is not None
        else request.tools
    )
    if len(tools) == 0 and has_tool_blocks(request.messages):
        return Error(
            TranslationError.of_unsupported(
                "tool history without tools: v1 raises unless modify_params injects the dummy tool"
            )
        )
    return _assemble(request, deps, thinking_json, tools, rf_choice)


def _gate(request: ChatRequest) -> TranslationError | None:
    if not p.is_anthropic_base(request.model):
        return TranslationError.of_unsupported(
            f"bedrock converse v2 serves Claude models only; {request.model} stays on v1"
        )
    if len(request.messages) == 0:
        return TranslationError.of_unsupported(
            "bedrock requires at least one non-system message: v1 raises unless modify_params"
        )
    return None


def _assemble(
    request: ChatRequest,
    deps: TranslationDeps,
    thinking_json: PlainJson | None,
    tools: Block[ToolDef],
    rf_choice: dict[str, PlainJson] | None,
) -> _SerializeResult:
    match _tool_choice_json(request, deps, thinking_json, rf_choice):
        case Result(tag="ok", ok=tool_choice):
            pass
        case Result(error=choice_err):
            return Error(choice_err)
    match _inference_config(request, deps, thinking_json):
        case Result(tag="ok", ok=inference_config):
            pass
        case Result(error=inference_err):
            return Error(inference_err)
    match _additional_fields(request, deps, thinking_json):
        case Result(tag="ok", ok=additional):
            pass
        case Result(error=additional_err):
            return Error(additional_err)
    messages = serialize_messages(request.messages)
    if isinstance(messages, TranslationError):
        return Error(messages)
    system = _system_json(request.system, request.model)
    body: Body = {
        "messages": messages,
        "inferenceConfig": inference_config,
        **({"additionalModelRequestFields": additional} if additional else {}),
        **({"system": system} if system is not None else {}),
    }
    return _with_tool_config(body, request, tools, tool_choice)


def _with_tool_config(
    body: Body,
    request: ChatRequest,
    tools: Block[ToolDef],
    tool_choice: dict[str, PlainJson] | None,
) -> _SerializeResult:
    if len(tools) == 0:
        return Ok(body)
    tool_blocks = serialize_tools(tools, request.model)
    if isinstance(tool_blocks, TranslationError):
        return Error(tool_blocks)
    tool_config: dict[str, PlainJson] = {
        "tools": tool_blocks,
        **({"toolChoice": tool_choice} if tool_choice is not None else {}),
    }
    return Ok({**body, "toolConfig": tool_config})


def _inference_config(
    request: ChatRequest, deps: TranslationDeps, thinking_json: PlainJson | None
) -> Result[dict[str, PlainJson], TranslationError]:
    max_tokens = p.max_tokens_json(request, thinking_json)
    config: dict[str, PlainJson] = {
        **({"maxTokens": max_tokens} if max_tokens is not None else {}),
        # converse keeps whitespace-only entries (drift item 7).
        **(
            {"stopSequences": list(request.params.stop)}
            if len(request.params.stop) > 0
            else {}
        ),
    }
    for param, key, value_opt in (
        ("temperature", "temperature", request.params.temperature),
        ("top_p", "topP", request.params.top_p),
    ):
        match value_opt:
            case Option(tag="some", some=value):
                pass
            case _:
                continue
        match anthropic_params.gate_sampling_param(request.model, param, value, deps):
            case Result(tag="ok", ok=Option(tag="some", some=kept)):
                config = {**config, key: kept}
            case Result(tag="ok", ok=_):
                continue
            case Result(error=err):
                return Error(err)
    return Ok(config)


def _additional_fields(
    request: ChatRequest, deps: TranslationDeps, thinking_json: PlainJson | None
) -> Result[dict[str, PlainJson], TranslationError]:
    top_k_field: dict[str, PlainJson]
    match request.params.top_k:
        case Option(tag="some", some=top_k):
            match anthropic_params.gate_sampling_param(
                request.model, "top_k", top_k, deps
            ):
                case Result(tag="ok", ok=Option(tag="some", some=kept)):
                    top_k_field = {"top_k": kept}
                case Result(tag="ok", ok=_):
                    top_k_field = {}
                case Result(error=err):
                    return Error(err)
        case _:
            top_k_field = {}
    parallel_field = request.parallel_tool_calls.map(
        lambda parallel: _parallel_tool_choice_field(request.model, parallel)
    ).default_value({})
    return Ok(
        {
            "stream": request.stream,
            **({"thinking": thinking_json} if thinking_json is not None else {}),
            **top_k_field,
            **parallel_field,
        }
    )


def _parallel_tool_choice_field(model: str, parallel: bool) -> dict[str, PlainJson]:
    if not p.is_claude_4_5_plus(model):
        return {}
    return {"tool_choice": {"disable_parallel_tool_use": not parallel}}


def _system_json(system: Block[SystemText], model: str) -> PlainJson | None:
    blocks: list[PlainJson] = []
    for text in system:
        blocks.append({"text": text.text})  # nosemgrep: translation-no-mutation
        point = system_cache_point(text.cache, model)
        if point is not None:
            blocks.append(point)  # nosemgrep: translation-no-mutation
    return blocks or None


_RfOutcome = tuple[ToolDef | None, dict[str, PlainJson] | None]


def _response_format_tool(
    request: ChatRequest, deps: TranslationDeps
) -> Result[_RfOutcome, TranslationError]:
    match request.response_format:
        case Option(tag="some", some=response_format):
            pass
        case _:
            return Ok((None, None))
    match response_format.tag:
        case "text" | "json_object":
            # json_object sets only the json_mode marker in v1, and converse
            # pops that marker before the transform seam: a body no-op.
            return Ok((None, None))
        case "json_schema":
            pass
        case _:
            assert_never(response_format.tag)
    if p.supports_native_structured_output(request.model, deps):
        return Error(
            TranslationError.of_unsupported(
                "native structured outputs (outputConfig.textFormat) stays on v1"
            )
        )
    conflicted = (
        len(request.tools) > 0
        or request.tool_choice.is_some()
        or request.parallel_tool_calls.is_some()
    )
    if conflicted:
        return Error(
            TranslationError.of_unsupported(
                "response_format json tool path combined with tools/tool_choice is body-order-dependent in v1"
            )
        )
    if request.stream:
        return Error(
            TranslationError.of_unsupported(
                "response_format json_schema with stream takes v1's fake_stream path"
            )
        )
    tool = response_format_tool(response_format.json_schema)
    if isinstance(tool, TranslationError):
        return Error(tool)
    forced: dict[str, PlainJson] | None = (
        None
        if anthropic_params.thinking_signaled(request)
        else {"tool": {"name": "json_tool_call"}}
    )
    return Ok((tool, forced))


def _tool_choice_json(
    request: ChatRequest,
    deps: TranslationDeps,
    thinking_json: PlainJson | None,
    rf_choice: dict[str, PlainJson] | None,
) -> Result[dict[str, PlainJson] | None, TranslationError]:
    choice_json: dict[str, PlainJson] | None
    match rf_choice, request.tool_choice:
        case None, Option(tag="some", some=choice):
            mapped = _map_tool_choice(choice, deps)
            if isinstance(mapped, TranslationError):
                return Error(mapped)
            choice_json = mapped
        case _:
            choice_json = rf_choice
    if (
        choice_json is not None
        and p.thinking_enabled_json(thinking_json)
        and ("any" in choice_json or "tool" in choice_json)
    ):
        # v1 downgrades forced tool use to auto when reasoning is enabled.
        return Ok({"auto": {}})
    return Ok(choice_json)


def _map_tool_choice(
    choice: ToolChoice, deps: TranslationDeps
) -> dict[str, PlainJson] | None | TranslationError:
    match choice.tag:
        case "auto":
            return {"auto": {}}
        case "required":
            return {"any": {}}
        case "none":
            if deps.drop_params:
                return None
            return TranslationError.of_unsupported(
                "tool_choice='none' on converse: v1 raises unless drop_params"
            )
        case "specific":
            return {"tool": {"name": make_valid_bedrock_name(choice.specific)}}
    assert_never(choice.tag)
