"""Round trip of an Anthropic server tool search through the Responses bridge.

Claude runs a tool search on the server. Its reply holds a ``server_tool_use`` block, a
``tool_search_tool_result`` block with ``tool_references``, and then the ``tool_use`` of the
tool it found. The reply side turns that into a ``tool_search_call`` item and a
``tool_search_output`` item. The request side turns both items, replayed in the next turn's
input, back into the same blocks, so Claude does not search again.
"""

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from pydantic import ConfigDict, TypeAdapter, ValidationError

from litellm.litellm_core_utils.prompt_templates.mid_conversation_system import is_anthropic_tool_search_use
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.responses.main import OutputToolSearchCall, OutputToolSearchOutput

from .custom_tools import openai_shaped_tool_call_item_id
from .tool_search import ANTHROPIC_TOOL_SEARCH_TOOL_NAME

__all__ = [
    "ServerSearchReplay",
    "build_tool_search_output",
    "holds_replayed_server_search",
    "provider_tool_results",
    "response_tool_results",
    "server_search_replay_of",
    "server_search_replays",
    "tool_reference_names",
    "tool_search_output_for_call",
    "without_replayed_server_searches",
]

ANTHROPIC_TOOL_SEARCH_RESULT_TYPE: Final = "tool_search_tool_result"
ANTHROPIC_TOOL_SEARCH_RESULT_CONTENT_TYPE: Final = "tool_search_tool_search_result"
ANTHROPIC_TOOL_REFERENCE_TYPE: Final = "tool_reference"

_OBJECTS: Final = TypeAdapter(tuple[object, ...])
_MESSAGE_FIELDS: Final = TypeAdapter(Mapping[str, object])


class _Reference(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    tool_name: str = ""


class _SearchResultContent(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    type: str = ""
    tool_references: tuple[_Reference, ...] = ()


class _SearchResultBlock(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    type: str = ""
    tool_use_id: str = ""
    content: _SearchResultContent | None = None


def _search_result_block(block: object) -> _SearchResultBlock | None:
    try:
        parsed: Final = _SearchResultBlock.model_validate(block)
    except ValidationError:
        return None
    if parsed.type != ANTHROPIC_TOOL_SEARCH_RESULT_TYPE or parsed.content is None:
        return None
    if parsed.content.type != ANTHROPIC_TOOL_SEARCH_RESULT_CONTENT_TYPE:
        return None
    return parsed


def tool_reference_names(tool_results: Iterable[object], call_id: str) -> tuple[str, ...] | None:
    """Names the search found for ``call_id``, or ``None`` when no search result answers it."""
    for block in tool_results:
        parsed = _search_result_block(block)
        if parsed is not None and parsed.content is not None and parsed.tool_use_id == call_id:
            return tuple(reference.tool_name for reference in parsed.content.tool_references if reference.tool_name)
    return None


def provider_tool_results(fields: object) -> tuple[object, ...]:
    if not isinstance(fields, Mapping):
        return ()
    results: Final = fields.get("tool_results")
    return tuple(results) if isinstance(results, (list, tuple)) else ()


def response_tool_results(chat_completion_response: object) -> tuple[object, ...]:
    """Server tool results a chat response carries, on its messages or in its hidden params."""
    tool_results: list[object] = []  # mutable-ok: collects results across choices
    for choice in getattr(chat_completion_response, "choices", None) or ():
        message = getattr(choice, "message", None)
        tool_results.extend(provider_tool_results(getattr(message, "provider_specific_fields", None)))
    hidden: Final = getattr(chat_completion_response, "_hidden_params", None)
    if isinstance(hidden, Mapping):
        tool_results.extend(provider_tool_results(hidden.get("provider_specific_fields")))
    return tuple(tool_results)


class _ToolFields(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    type: str = ""
    name: str = ""
    tools: tuple[Mapping[str, object], ...] = ()


def _tool_fields(tool: object) -> _ToolFields | None:
    try:
        return _ToolFields.model_validate(tool)
    except ValidationError:
        return None


def _is_namespace(fields: _ToolFields) -> bool:
    return fields.type == "namespace"


def _names_of(tool: object) -> tuple[str, ...]:
    """Chat-side names of a Responses tool: ``name``, or ``namespace__name`` for a namespace member."""
    fields: Final = _tool_fields(tool)
    if fields is None:
        return ()
    if not _is_namespace(fields):
        return (fields.name,) if fields.name else ()
    return tuple(f"{fields.name}__{member.get('name') or ''}" for member in fields.tools)


def _tools_named(request_tools: Sequence[object] | None, names: frozenset[str]) -> list[dict[str, object]]:
    found: Final[list[dict[str, object]]] = []  # mutable-ok: collects matches in request order
    for tool in request_tools or ():
        fields = _tool_fields(tool)
        if fields is None or not isinstance(tool, Mapping):
            continue
        if not _is_namespace(fields):
            if fields.name in names:
                found.append(dict(tool))
            continue
        members = tuple(member for member in fields.tools if f"{fields.name}__{member.get('name') or ''}" in names)
        if members:
            found.append({**tool, "tools": list(members)})
    return found


def build_tool_search_output(
    call_id: str, names: Sequence[str], request_tools: Sequence[object] | None
) -> OutputToolSearchOutput:
    return OutputToolSearchOutput(
        id=openai_shaped_tool_call_item_id("tool_search_output", call_id),
        call_id=call_id,
        execution="server",
        status="completed",
        tools=_tools_named(request_tools, frozenset(names)),
        type="tool_search_output",
    )


def tool_search_output_for_call(
    call: OutputToolSearchCall, tool_results: Iterable[object], request_tools: Sequence[object] | None
) -> OutputToolSearchOutput | None:
    if call.execution != "server" or not call.call_id:
        return None
    names: Final = tool_reference_names(tool_results, call.call_id)
    return None if names is None else build_tool_search_output(call.call_id, names, request_tools)


class _ReplayedSearchItem(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    type: str = ""
    execution: str | None = None
    call_id: str | None = None
    arguments: object = None
    tools: tuple[Mapping[str, object], ...] = ()


def _server_search_item(item: object, item_type: str) -> _ReplayedSearchItem | None:
    try:
        parsed: Final = _ReplayedSearchItem.model_validate(item)
    except ValidationError:
        return None
    if parsed.type != item_type or parsed.execution == "client" or not parsed.call_id:
        return None
    return parsed


def _arguments_json(arguments: object) -> str:
    return arguments if isinstance(arguments, str) else json.dumps(arguments or {})


@dataclass(frozen=True, slots=True)
class ServerSearchReplay:
    call_id: str
    arguments_json: str
    tool_names: tuple[str, ...]

    @property
    def tool_call(self) -> dict[str, object]:
        return {
            "id": self.call_id,
            "type": "function",
            "function": {"name": ANTHROPIC_TOOL_SEARCH_TOOL_NAME, "arguments": self.arguments_json},
            "index": 0,
        }

    @property
    def tool_result(self) -> dict[str, object]:
        return {
            "type": ANTHROPIC_TOOL_SEARCH_RESULT_TYPE,
            "tool_use_id": self.call_id,
            "content": {
                "type": ANTHROPIC_TOOL_SEARCH_RESULT_CONTENT_TYPE,
                "tool_references": [
                    {"type": ANTHROPIC_TOOL_REFERENCE_TYPE, "tool_name": name} for name in self.tool_names
                ],
            },
        }

    def assistant_message(self) -> dict[str, object]:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [self.tool_call],
            "provider_specific_fields": {"tool_results": [self.tool_result]},
        }

    def added_to(self, message: Mapping[str, object]) -> dict[str, object]:
        """``message`` (an assistant turn with tool calls) plus this search, after its other calls."""
        calls: Final = message.get("tool_calls")
        fields: Final = message.get("provider_specific_fields")
        previous_results: Final = provider_tool_results(fields)
        return {
            **message,
            "tool_calls": [*(calls if isinstance(calls, (list, tuple)) else ()), self.tool_call],
            "provider_specific_fields": {
                **(fields if isinstance(fields, Mapping) else {}),
                "tool_results": [*previous_results, self.tool_result],
            },
        }


def server_search_replays(input: object) -> Mapping[str, ServerSearchReplay]:
    """Server tool searches an input replays whole: a ``tool_search_call`` and its ``tool_search_output``.

    A search call that no output answers stays out. Claude gets no ``server_tool_use`` block
    it has no result for.
    """
    if not isinstance(input, (list, tuple)):
        return {}
    try:
        items: Final = _OBJECTS.validate_python(input)
    except ValidationError:
        return {}
    outputs: Final = {
        parsed.call_id: parsed
        for item in items
        if (parsed := _server_search_item(item, "tool_search_output")) is not None and parsed.call_id
    }
    replays: Final[dict[str, ServerSearchReplay]] = {}  # mutable-ok: indexes replays by call id
    for item in items:
        call = _server_search_item(item, "tool_search_call")
        if call is None or call.call_id is None or call.call_id in replays:
            continue
        output = outputs.get(call.call_id)
        if output is None or not is_anthropic_tool_search_use(call.call_id, ANTHROPIC_TOOL_SEARCH_TOOL_NAME):
            continue
        replays[call.call_id] = ServerSearchReplay(
            call_id=call.call_id,
            arguments_json=_arguments_json(call.arguments),
            tool_names=tuple(name for tool in output.tools for name in _names_of(tool)),
        )
    return replays


def server_search_replay_of(item: object, replays: Mapping[str, ServerSearchReplay]) -> ServerSearchReplay | None:
    """The replay a ``tool_search_call`` input item stands for, if its output was replayed too."""
    call: Final = _server_search_item(item, "tool_search_call")
    return None if call is None or call.call_id is None else replays.get(call.call_id)


def _replayed_search_ids(message: Mapping[str, object]) -> frozenset[str]:
    return frozenset(
        block.tool_use_id
        for result in provider_tool_results(message.get("provider_specific_fields"))
        if (block := _search_result_block(result)) is not None
    )


def _call_id_of(tool_call: object) -> str | None:
    call_id: Final = tool_call.get("id") if isinstance(tool_call, Mapping) else getattr(tool_call, "id", None)
    return call_id if isinstance(call_id, str) else None


def _without_replayed_search(message: Mapping[str, object]) -> Mapping[str, object] | None:
    search_ids: Final = _replayed_search_ids(message)
    if not search_ids:
        return message
    calls: Final = message.get("tool_calls")
    kept_calls: Final = [call for call in calls or () if _call_id_of(call) not in search_ids]
    fields: Final = message.get("provider_specific_fields")
    kept_fields: Final = {
        key: value for key, value in (fields.items() if isinstance(fields, Mapping) else ()) if key != "tool_results"
    }
    rest: Final = {
        key: value for key, value in message.items() if key not in ("tool_calls", "provider_specific_fields")
    }
    if not kept_calls and not message.get("content"):
        return None
    return {
        **rest,
        **({"tool_calls": kept_calls} if kept_calls else {}),
        **({"provider_specific_fields": kept_fields} if kept_fields else {}),
    }


def holds_replayed_server_search(messages: Iterable[object]) -> bool:
    return any(
        isinstance(message, Mapping) and bool(_replayed_search_ids(_MESSAGE_FIELDS.validate_python(message)))
        for message in messages
    )


def without_replayed_server_searches(messages: Sequence[object]) -> list[object]:
    """Messages without the server searches this module replayed, for a route that builds no Anthropic request."""
    kept: Final[list[object]] = []  # mutable-ok: collects the kept messages in order
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            kept.append(message)
            continue
        stripped = _without_replayed_search(_MESSAGE_FIELDS.validate_python(message))
        if stripped is not None:
            kept.append(stripped)
    return kept
