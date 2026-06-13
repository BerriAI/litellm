"""Serialize the IR into an ollama ``/api/chat`` request body.

Mirrors ``OllamaChatConfig.map_openai_params`` + ``transform_request``
(probed in-process at HEAD): the body is ``{model, messages, options,
stream}`` plus top-level ``format``/``tools``/``think``. The openai params
land in ``options`` under ollama names (max_tokens/mct -> ``num_predict``,
last write wins — the raw guard rejects both keys together), ``top_k`` rides
``options`` via v1's provider-native passthrough, ``stop`` rides verbatim,
and ``stream`` is ALWAYS a body key (default false).

Message munge, per v1's whitelist transform over the openai wire messages
(the shared ``serialize_messages`` inverse is the proven reconstruction of
exactly those messages):

- ``content`` is ALWAYS a string: lists flatten to the concatenation of
  their truthy text parts, ``None`` becomes ``""``;
- a STRING content matching v1's ``<think>``/``<thinking>``/
  ``<budget:thinking>`` open+close regex emits ``thinking`` = the inner text
  while the content stays the FULL tagged string (v1 reads the regex result
  for ``thinking`` only and flattens the ORIGINAL content — probed; the
  remainder group is discarded). List content is never think-parsed (v1's
  string-only fork; the raw guard keeps single-text lists on v1);
- ``images`` is attached to EVERY message (``[]`` included): image parts'
  urls with ``data:...;base64,`` prefixes are stripped to the bare base64
  payload, other urls ride verbatim;
- assistant ``tool_calls`` become ``[{function: {name, arguments:
  <parsed dict>}}]`` — the wire ``id`` and ``type`` are DROPPED and the
  argument string is ``json.loads``-ed (a blank argument string raises
  json.JSONDecodeError in v1 — typed fallback here);
- ``tool_call_id`` is forwarded; message ``name`` is dropped (v1 parity).

``response_format`` json_object -> ``format: "json"``; json_schema -> the
bare schema dict (name/strict dropped; a FALSY schema is dropped entirely —
v1's truthiness gate). ``reasoning_effort`` -> ``think``: verbatim for
``gpt-oss*`` models, else the boolean ``value in {low, medium, high}``.
``tool_choice`` is silently dropped (v1 pops it — hang avoidance) and the
caller's tools ride VERBATIM (openai shape, ``strict`` included).

Ambient ``OllamaChatConfig`` class-attr defaults merge into ``options`` in
v1 (the groq class-attr precedent) — module state v2 cannot see; the future
completion() fork MUST fall back to v1 when ``OllamaChatConfig.get_config()``
is non-empty (CLAUDE.md fork obligation).
"""

from __future__ import annotations

import json
import re

from expression import Error, Ok, Option, Result

from ...deps import TranslationDeps
from ...errors import TranslationError
from ...ir import Body, ChatRequest, PlainJson, ResponseFormat
from ..openai_compat.serialize import assemble_body
from . import params as p

_SerializeResult = Result[Body, TranslationError]

THINK_TAG_RE = re.compile(
    r"<(?:think|thinking|budget:thinking)>(.*?)</(?:think|thinking|budget:thinking)>(.*)",
    re.DOTALL,
)
"""The ONE in-package mirror of v1's ``_parse_content_for_reasoning`` regex
(litellm_core_utils/prompt_templates/common_utils.py); the request gate's
mirror test pins it against v1's source."""


def serialize_request(request: ChatRequest, deps: TranslationDeps) -> _SerializeResult:
    reason = p.unsupported_params(request, deps)
    if reason is not None:
        return Error(TranslationError.of_unsupported(reason))
    match assemble_body(request):
        case Result(tag="ok", ok=assembled):
            pass
        case Result(error=err):
            return Error(err)
    messages = _munged_messages(assembled.get("messages"))
    if isinstance(messages, TranslationError):
        return Error(messages)
    body: Body = {
        "model": request.model,
        "messages": messages,
        "options": _options(request),
        "stream": request.stream,
        **_format_field(request.response_format),
        **_tools_field(assembled.get("tools")),
        **_think_field(request),
    }
    return Ok(body)


def _options(request: ChatRequest) -> dict[str, PlainJson]:
    params = request.params
    out: dict[str, PlainJson] = {}
    num_predict = params.max_completion_tokens.default_value(
        params.max_tokens.default_value(None)
    )
    fields: tuple[tuple[str, PlainJson | None], ...] = (
        ("num_predict", num_predict),
        ("temperature", params.temperature.default_value(None)),
        ("top_p", params.top_p.default_value(None)),
        ("stop", list(params.stop) if len(params.stop) > 0 else None),
        ("top_k", params.top_k.default_value(None)),
    )
    for key, value in fields:
        if value is not None:
            out = {**out, key: value}
    return out


def _format_field(response_format: Option[ResponseFormat]) -> dict[str, PlainJson]:
    rf = response_format.default_value(None)
    if rf is None:
        return {}
    match rf.tag:
        case "json_object":
            return {"format": "json"}
        case "json_schema":
            schema = rf.json_schema.schema.value
            # v1's truthiness gate: a falsy schema never reaches the wire
            return {"format": schema} if schema else {}
        case _:
            return {}  # "text" is dropped entirely (probed)


def _tools_field(tools: PlainJson | None) -> dict[str, PlainJson]:
    if tools is None:
        return {}
    return {"tools": tools}


def _think_field(request: ChatRequest) -> dict[str, PlainJson]:
    effort = request.reasoning_effort.default_value(None)
    if effort is None:
        return {}
    if request.model.startswith("gpt-oss"):
        return {"think": effort}
    return {"think": effort in ("low", "medium", "high")}


def _munged_messages(messages: PlainJson | None) -> list[PlainJson] | TranslationError:
    if not isinstance(messages, list):
        return TranslationError.of_unsupported(
            "ollama_chat serializer received a non-list message assembly; wiring bug"
        )
    munged = [_munge_message(message) for message in messages]
    for entry in munged:
        if isinstance(entry, TranslationError):
            return entry
    return [entry for entry in munged if not isinstance(entry, TranslationError)]


def _munge_message(message: PlainJson) -> PlainJson | TranslationError:
    if not isinstance(message, dict):
        return TranslationError.of_unsupported(
            "ollama_chat serializer received a non-object wire message; wiring bug"
        )
    content = message.get("content")
    tool_calls = _ollama_tool_calls(message.get("tool_calls"))
    if isinstance(tool_calls, TranslationError):
        return tool_calls
    out: dict[str, PlainJson] = {"role": message.get("role")}
    thinking = _think_tag_reasoning(content)
    if thinking is not None:
        out = {**out, "thinking": thinking}
    out = {**out, "content": _flattened_content(content), "images": _images(content)}
    if tool_calls is not None:
        out = {**out, "tool_calls": tool_calls}
    tool_call_id = message.get("tool_call_id")
    if tool_call_id is not None:
        out = {**out, "tool_call_id": tool_call_id}
    return out


def _think_tag_reasoning(content: PlainJson) -> str | None:
    if not isinstance(content, str):
        return None  # v1 think-parses STRING content only
    matched = THINK_TAG_RE.match(content)
    return matched.group(1) if matched else None


def _flattened_content(content: PlainJson) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""  # v1's convert_content_list_to_str returns "" for None
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and text:
            parts = [*parts, text]
    return "".join(parts)


def _images(content: PlainJson) -> list[PlainJson]:
    if not isinstance(content, list):
        return []
    out: list[PlainJson] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        image_url = item.get("image_url")
        if isinstance(image_url, str) and image_url:
            out = [*out, _base64_payload(image_url)]
        elif isinstance(image_url, dict) and "url" in image_url:
            url = image_url.get("url")
            if isinstance(url, str):
                out = [*out, _base64_payload(url)]
    return out


def _base64_payload(url: str) -> str:
    if url.startswith("data:") and ";base64," in url:
        return url.split(";base64,", 1)[1]
    return url


def _ollama_tool_calls(
    tool_calls: PlainJson,
) -> list[PlainJson] | None | TranslationError:
    if tool_calls is None:
        return None
    if not isinstance(tool_calls, list):
        return TranslationError.of_unsupported(
            "ollama_chat serializer received non-list tool_calls; wiring bug"
        )
    out: list[PlainJson] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            return TranslationError.of_unsupported(
                "ollama_chat serializer received a non-object tool_call; wiring bug"
            )
        function = call.get("function")
        function_map = function if isinstance(function, dict) else {}
        arguments_raw = function_map.get("arguments")
        try:
            arguments: PlainJson = (
                json.loads(arguments_raw) if isinstance(arguments_raw, str) else {}
            )
        except ValueError:
            return TranslationError.of_unsupported(
                "tool_call arguments that json.loads rejects (blank string "
                "included): v1's ollama_chat munge raises json.JSONDecodeError"
            )
        name = function_map.get("name")
        out = [
            *out,
            {"function": {"name": name if name else "", "arguments": arguments}},
        ]
    return out
