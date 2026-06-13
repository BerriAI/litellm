"""Serialize the IR into a databricks ``/serving-endpoints`` chat body.

databricks' wire is openai-chat-shaped: ``DatabricksConfig``'s MRO routes
``transform_request`` to ``OpenAIGPTConfig`` (the anthropic serializer NEVER
runs), while the anthropic planes contribute pure param mappers. The body is
``{model, messages, **mapped_params}`` with ``stream`` ALWAYS materialized
(default False). Probed in-process at HEAD; the central fork is
``"claude" in model`` (DB-R1, ``params.is_claude``).

The served set (the narrower wave-3 split; everything else is a typed
fallback the params gate / guard reports so v1 serves its own behavior):

- messages: the shared ``serialize_messages`` inverse (the guard already falls
  back on cache_control, whitespace-only content, and USER message ``name`` —
  the cases v1's munge handles losslessly or keeps; assistant/tool names are
  stripped by v1 == the IR drop);
- temperature/top_p/stop verbatim; ``top_k`` verbatim TOP-LEVEL (DB-R drift:
  researcher-5 said unsupported, the probe serves it both arms);
- ``max_tokens``: mct is ALWAYS renamed to ``max_tokens`` (v1 never re-emits
  max_completion_tokens), then the thinking max-bump (``tools.thinking_max_tokens``)
  overrides ONLY when thinking carries a budget and the caller gave none;
- tools: NON-claude verbatim (the shared assembly), claude via the
  ``tools.claude_tools`` round-trip (drops strict/$schema/cache_control);
- tool_choice verbatim;
- ``thinking``: passthrough as ``{type:enabled[,budget_tokens]}`` for ANY model
  (the bump rides max_tokens);
- ``reasoning_effort``: served ONLY on NON-claude WITH max_tokens (rides the
  wire verbatim — one line); claude reasoning_effort and non-claude-without-
  max_tokens are params-gate fallbacks (the thinking machinery / the DB-R3
  KeyError crash);
- ``response_format``: NON-claude verbatim (claude is a params-gate fallback —
  the json_tool_call machinery and the DB-R2 json_object silent drop).

The M2M token POST and the WorkspaceClient URL synthesis live in the ENVELOPE
(v1's own resolution, the watsonx-IAM rule); this serializer reads no env and
triggers no network (semgrep-enforced purity).
"""

from __future__ import annotations

from expression import Error, Ok, Option, Result

from ...deps import TranslationDeps
from ...errors import TranslationError
from ...ir import Body, ChatRequest, PlainJson, ToolDef
from ..anthropic.params import thinking_json
from ..openai_compat.messages import serialize_messages
from ..openai_compat.serialize import _response_format_json
from . import params as p
from . import tools as t

_SerializeResult = Result[Body, TranslationError]


def serialize_request(request: ChatRequest, deps: TranslationDeps) -> _SerializeResult:
    reason = p.unsupported_params(request)
    if reason is not None:
        return Error(TranslationError.of_unsupported(reason))
    messages = serialize_messages(request)
    if isinstance(messages, TranslationError):
        return Error(messages)
    tools = _tools(request)
    if isinstance(tools, TranslationError):
        return Error(tools)
    body: Body = {
        "model": request.model,
        "messages": messages,
        "stream": request.stream,
        **_sampling(request),
        **_tools_field(tools),
        **_tool_choice_field(request),
        **_thinking_field(request),
        **_reasoning_effort_field(request),
        **_response_format_field(request),
        **_max_tokens_field(request),
    }
    return Ok(body)


def _sampling(request: ChatRequest) -> dict[str, PlainJson]:
    params = request.params
    fields: tuple[tuple[str, PlainJson | None], ...] = (
        ("temperature", params.temperature.default_value(None)),
        ("top_p", params.top_p.default_value(None)),
        ("top_k", params.top_k.default_value(None)),
        ("stop", list(params.stop) if len(params.stop) > 0 else None),
    )
    return {key: value for key, value in fields if value is not None}


def _max_tokens_field(request: ChatRequest) -> dict[str, PlainJson]:
    caller = request.params.max_completion_tokens.default_value(
        request.params.max_tokens.default_value(None)
    )
    match request.thinking:
        case Option(tag="some", some=thinking):
            bumped = t.thinking_max_tokens(thinking, caller)
        case _:
            bumped = None
    value = bumped if bumped is not None else caller
    return {"max_tokens": value} if value is not None else {}


def _tools(request: ChatRequest) -> list[PlainJson] | None | TranslationError:
    if len(request.tools) == 0:
        return None
    if p.is_claude(request.model):
        return t.claude_tools(request.tools)
    return [_verbatim_tool(tool) for tool in request.tools]


def _verbatim_tool(tool: ToolDef) -> PlainJson:
    function: dict[str, PlainJson] = {"name": tool.name}
    match tool.description:
        case Option(tag="some", some=description):
            function = {**function, "description": description}
        case _:
            pass
    match tool.parameters:
        case Option(tag="some", some=parameters):
            function = {**function, "parameters": parameters.value}
        case _:
            pass
    match tool.strict:
        case Option(tag="some", some=strict):
            function = {**function, "strict": strict}
        case _:
            pass
    return {"type": "function", "function": function}


def _tools_field(tools: list[PlainJson] | None) -> dict[str, PlainJson]:
    return {"tools": tools} if tools is not None else {}


def _tool_choice_field(request: ChatRequest) -> dict[str, PlainJson]:
    match request.tool_choice:
        case Option(tag="some", some=choice):
            pass
        case _:
            return {}
    match choice.tag:
        case "auto":
            return {"tool_choice": "auto"}
        case "required":
            return {"tool_choice": "required"}
        case "none":
            return {"tool_choice": "none"}
        case "specific":
            return {
                "tool_choice": {
                    "type": "function",
                    "function": {"name": choice.specific},
                }
            }
    return {}


def _thinking_field(request: ChatRequest) -> dict[str, PlainJson]:
    match request.thinking:
        case Option(tag="some", some=thinking):
            return {"thinking": thinking_json(thinking)}
        case _:
            return {}


def _reasoning_effort_field(request: ChatRequest) -> dict[str, PlainJson]:
    match request.reasoning_effort:
        case Option(tag="some", some=effort):
            return {"reasoning_effort": effort}
        case _:
            return {}


def _response_format_field(request: ChatRequest) -> dict[str, PlainJson]:
    """NON-claude only (the params gate falls back on claude). The wire shape
    is the openai one verbatim — REUSE the shared builder (v1 passes the
    caller's response_format through unchanged on non-claude models)."""
    value = _response_format_json(request.response_format)
    return {"response_format": value} if value is not None else {}
