"""Serialize the IR into a snowflake Cortex ``inference:complete`` body.

v1's ``transform_request`` does NOT call super(): messages go out VERBATIM
(no content-list flatten, no base image transforms, no name strip — the
openai guard's full-name fallback covers the IR's name drop), ``stream``
is ALWAYS in the body (``optional_params.pop("stream", None) or False`` —
so absent and explicit-false are the SAME wire byte, and the family's
explicit-stream-false guard arm is deliberately NOT composed here: v2
serves it), tools are rewritten to Snowflake's ``tool_spec`` shape and
tool_choice to its object shape (both pure, verified in-process at HEAD):

- tool -> ``{"tool_spec": {"type": "generic", "name", "input_schema"
  (parameters, default {"type":"object","properties":{}}), "description"?}}``
- tool_choice ``"auto"``/``"none"`` -> ``{"type": <same>}``;
  ``"required"`` -> ``{"type": "any"}``; the dict-function form ->
  ``{"type": "tool", "name": [<fn>]}`` (an ARRAY).

``top_k`` rides verbatim (the non-compat top-level passthrough). The
account_id -> URL synthesis and the KEYPAIR_JWT/PAT auth headers are pure
envelope (validate_environment sets headers only — researcher-4's
correction to researcher-3 stands at HEAD). The response model is
``snowflake/{WIRE model}`` — response.py owns that prefix.
"""

from __future__ import annotations

from expression import Error, Result

from ...deps import TranslationDeps
from ...errors import TranslationError
from ...ir import Body, ChatRequest, PlainJson
from ..openai_compat.serialize import assemble_body
from . import params as p

_SerializeResult = Result[Body, TranslationError]


def serialize_request(request: ChatRequest, deps: TranslationDeps) -> _SerializeResult:
    reason = p.unsupported_params(request, deps)
    if reason is not None:
        return Error(TranslationError.of_unsupported(reason))
    return assemble_body(request).map(
        lambda body: _with_snowflake_deltas(body, request)
    )


def _with_snowflake_deltas(body: Body, request: ChatRequest) -> Body:
    patched: dict[str, PlainJson] = {**body, "stream": request.stream}
    tools = patched.get("tools")
    if isinstance(tools, list):
        patched = {**patched, "tools": [_tool_spec(tool) for tool in tools]}
    tool_choice = patched.get("tool_choice")
    if tool_choice is not None:
        patched = {**patched, "tool_choice": _tool_choice_object(tool_choice)}
    top_k = request.params.top_k.default_value(None)
    if top_k is not None:
        patched = {**patched, "top_k": top_k}
    return patched


def _tool_spec(tool: PlainJson) -> PlainJson:
    function = tool.get("function") if isinstance(tool, dict) else None
    if not isinstance(function, dict):
        return tool  # unreachable: the inbound parse admits function tools only
    spec: dict[str, PlainJson] = {
        "type": "generic",
        "name": function.get("name"),
        "input_schema": function.get(
            "parameters", {"type": "object", "properties": {}}
        ),
    }
    description = function.get("description")
    if description is not None:
        spec = {**spec, "description": description}
    return {"tool_spec": spec}


def _tool_choice_object(tool_choice: PlainJson) -> PlainJson:
    if isinstance(tool_choice, str):
        mapped = {"auto": "auto", "required": "any", "none": "none"}.get(
            tool_choice, tool_choice
        )
        return {"type": mapped}
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        function = tool_choice.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        if name:
            return {"type": "tool", "name": [name]}
    return tool_choice  # v1's trailing verbatim arm (unreachable shapes)
