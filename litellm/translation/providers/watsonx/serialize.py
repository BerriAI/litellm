"""Serialize the IR into a watsonx ``/ml/v1/text/chat`` request body.

v1's chain is ``IBMWatsonXChatConfig.map_openai_params`` then the
``WatsonXChatHandler`` -> ``OpenAILikeChatHandler`` body assembly
``{"model": model_id, "messages": <base-transformed>, **optional_params}``
with the ``_prepare_payload`` injection merged into optional_params first
(probed in-process at HEAD). The body is the openai_compat assembly with:

- ``stream`` ALWAYS present: the handler pops it and re-adds
  ``stream or False`` unconditionally, so absent and explicit-false both
  serialize ``false``;
- ``model_id``: the wire model duplicated next to ``model`` (the handler
  passes ``watsonx_auth_payload["model_id"]`` as the data model AND the key
  rides optional_params);
- ``project_id`` (or ``space_id`` when no project) from deps —
  ``_prepare_payload`` injects exactly one;
- tools: ``additionalProperties`` removed ONLY where its value is False,
  and ``strict`` removed at EVERY depth (``_remove_additional_properties``
  + ``_remove_strict_from_schema``, both uncapped in v1; the shared
  recursion cap here returns the subtree untouched past depth 16 — the
  xai ``_strip_cache`` precedent);
- ``tool_choice``: the string options auto/none/required move to
  ``tool_choice_option``; the dict form rides ``tool_choice`` verbatim;
- ``response_format``/``reasoning_effort`` verbatim (both in the list;
  json_mode is NEVER set for watsonx, so the OpenAILike json_mode
  machinery stays dormant — probed).
"""

from __future__ import annotations

from expression import Error, Result
from typing_extensions import assert_never

from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH

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
        lambda body: _with_watsonx_deltas(body, request, deps)
    )


def _with_watsonx_deltas(
    body: Body, request: ChatRequest, deps: TranslationDeps
) -> Body:
    reshaped: dict[str, PlainJson] = {}
    for key, value in body.items():
        if key == "tools":
            reshaped = {**reshaped, "tools": _cleaned_tools(value, 0)}
        elif key == "tool_choice":
            reshaped = {**reshaped, **_tool_choice_fields(request, value)}
        elif key == "stream":
            pass  # re-emitted unconditionally below
        else:
            reshaped = {**reshaped, key: value}
    effort = request.reasoning_effort.default_value(None)
    if effort is not None:
        # in the supported list unconditionally; the base map passes it
        # verbatim (assemble_body treats it as a typed-fallback param, so
        # the emission is a watsonx delta — the xai precedent)
        reshaped = {**reshaped, "reasoning_effort": effort}
    payload: dict[str, PlainJson] = {"model_id": request.model}
    if deps.watsonx_project_id is not None:
        payload = {**payload, "project_id": deps.watsonx_project_id}
    else:
        payload = {**payload, "space_id": deps.watsonx_space_id}
    return {**reshaped, "stream": request.stream, **payload}


def _tool_choice_fields(
    request: ChatRequest, serialized: PlainJson
) -> dict[str, PlainJson]:
    choice = request.tool_choice.default_value(None)
    if choice is None:
        return {}
    match choice.tag:
        case "auto" | "required" | "none":
            return {"tool_choice_option": serialized}
        case "specific":
            return {"tool_choice": serialized}
    assert_never(choice.tag)


def _cleaned_tools(value: PlainJson, depth: int) -> PlainJson:
    """Mirror v1's two uncapped sweeps in one pass: drop
    ``additionalProperties`` only when False, drop ``strict`` everywhere."""
    if depth > DEFAULT_MAX_RECURSE_DEPTH:
        return value
    if isinstance(value, dict):
        return {
            key: _cleaned_tools(item, depth + 1)
            for key, item in value.items()
            if key != "strict" and not (key == "additionalProperties" and item is False)
        }
    if isinstance(value, list):
        return [_cleaned_tools(item, depth + 1) for item in value]
    return value
