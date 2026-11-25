from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import reduce
from itertools import chain, groupby
from types import MappingProxyType
from typing import Final

from mcp.types import Tool as MCPTool
from pydantic import BaseModel, TypeAdapter, ValidationError

from litellm._uuid import uuid
from litellm.proxy.mcp_server.utils import strip_known_server_prefix
from litellm.responses.mcp.direct_url_claims import DirectListing, server_names
from litellm.responses.mcp.mcp_streaming_iterator import output_items
from litellm.responses.mcp.tool_precheck import LITELLM_PROXY_MCP_SERVER_URL_PREFIX
from litellm.types.llms.base import BaseLiteLLMOpenAIResponseObject
from litellm.types.llms.openai import (
    MCPCallFailedEvent,
    MCPListToolsCompletedEvent,
    MCPListToolsInProgressEvent,
    OutputItemAddedEvent,
    OutputItemDoneEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
    ResponsesAPIStreamingResponse,
)
from litellm.types.mcp_server.mcp_server_manager import MCPServer

_TERMINAL_EVENTS: Final = frozenset(
    {
        ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
        ResponsesAPIStreamEvents.RESPONSE_INCOMPLETE,
        ResponsesAPIStreamEvents.RESPONSE_FAILED,
    }
)
_ITEM_FIELDS: Final = TypeAdapter(Mapping[str, object])
_MESSAGE_PARTS: Final = TypeAdapter(tuple[object, ...])
_OPENING_EVENTS: Final = frozenset(
    {ResponsesAPIStreamEvents.RESPONSE_CREATED, ResponsesAPIStreamEvents.RESPONSE_IN_PROGRESS}
)


@dataclass(frozen=True, slots=True)
class GatewayToolName:
    server_label: str
    name: str


@dataclass(frozen=True, slots=True)
class OpenAIMCPItems:
    """How the tools the gateway runs read in OpenAI's item shapes: one mcp_list_tools item per server, and the
    caller's server_label and bare tool name for every tool name the model sees. A held server gets neither, since its
    calls go back to the caller as function_call items the way they do without a claim."""

    list_items: tuple[Mapping[str, object], ...]
    names: Mapping[str, GatewayToolName]

    def runs_every_call(self, tool_names: Iterable[str | None]) -> bool:
        return all(name in self.names for name in tool_names)


@dataclass(frozen=True, slots=True)
class HeldReferences:
    """The caller's `litellm_proxy` references without `require_approval: "never"`. While any is present, a registered
    tool runs on the gateway only when the gateway claimed its server and no such reference names it, so the model's
    other calls go back to the caller the way they do without a claim. A reference whose name is no registered server,
    such as a toolset or an access group, can stand for any registered server, so it holds them all."""

    present: bool
    whole_gateway: bool
    server_names: frozenset[str]

    def lets_run(self, server_keys: Sequence[str], server_labels: Mapping[str, str]) -> bool:
        if not self.present:
            return True
        names_server: Final = self.whole_gateway or any(key.lower() in self.server_names for key in server_keys)
        return not names_server and any(key in server_labels for key in server_keys)


def held_references(
    gateway_tools: Sequence[Mapping[str, object]], server_for: Callable[[str], MCPServer | None]
) -> HeldReferences:
    held_urls: Final = tuple(
        str(tool.get("server_url") or "") for tool in gateway_tools if tool.get("require_approval") != "never"
    )
    held_names: Final = tuple(
        url.split("/")[-1] for url in held_urls if url.startswith(LITELLM_PROXY_MCP_SERVER_URL_PREFIX)
    )
    return HeldReferences(
        present=bool(held_urls),
        whole_gateway=len(held_names) < len(held_urls) or any(server_for(name) is None for name in held_names),
        server_names=frozenset(name.lower() for name in held_names),
    )


@dataclass(frozen=True, slots=True)
class CallerLabels:
    """The server_label the caller gave each server, which OpenAI repeats on every mcp_list_tools and mcp_call item so
    a client can match the items to its tools. A server takes the label of the first reference or claimed URL naming
    it. A server listed only through a reference that names no registered server, such as a toolset, an access group
    or the whole gateway, takes the label of the first such reference."""

    by_server: Mapping[str, str]
    otherwise: str | None

    def label_for(self, server_keys: Sequence[str]) -> str | None:
        return next(
            (self.by_server[key.lower()] for key in server_keys if key.lower() in self.by_server), self.otherwise
        )


def _referenced_server(tool: Mapping[str, object], server_for: Callable[[str], MCPServer | None]) -> MCPServer | None:
    url: Final = str(tool.get("server_url") or "")
    return server_for(url.split("/")[-1]) if url.startswith(LITELLM_PROXY_MCP_SERVER_URL_PREFIX) else None


def caller_labels(
    gateway_tools: Sequence[Mapping[str, object]], server_for: Callable[[str], MCPServer | None]
) -> CallerLabels:
    """The labels of the gateway tools, where each claimed URL already reads as a reference to its server."""
    references: Final = tuple(
        (str(tool.get("server_label") or "unlabeled"), _referenced_server(tool, server_for)) for tool in gateway_tools
    )
    named: Final = chain.from_iterable(
        ((name.lower(), label) for name in server_names(server)) for label, server in references if server is not None
    )
    return CallerLabels(
        by_server=MappingProxyType(dict(reversed(tuple(named)))),
        otherwise=next((label for label, server in references if server is None), None),
    )


@dataclass(frozen=True, slots=True)
class _RegisteredTool:
    tool: MCPTool
    name: GatewayToolName
    server_keys: tuple[str, ...]


def _field(item: object, key: str) -> object:
    if isinstance(item, BaseModel):
        return getattr(item, key, None)
    try:
        return _ITEM_FIELDS.validate_python(item, strict=True).get(key)
    except ValidationError:
        return getattr(item, key, None)


def _text(item: object, key: str) -> str | None:
    value: Final = _field(item, key)
    return value if isinstance(value, str) else None


def _registered_tool(
    tool: MCPTool, server_key: str, labels: CallerLabels, server_for: Callable[[str], MCPServer | None]
) -> _RegisteredTool:
    server: Final = server_for(server_key)
    keys: Final = (server_key, *(server_names(server) if server is not None else ()))
    return _RegisteredTool(
        tool=tool,
        name=GatewayToolName(
            server_label=labels.label_for(keys) or server_key, name=strip_known_server_prefix(tool.name, server)
        ),
        server_keys=keys,
    )


def _shared_names(names: Iterable[GatewayToolName]) -> frozenset[GatewayToolName]:
    ordered: Final = sorted(names, key=lambda name: (name.server_label, name.name))
    return frozenset(name for name, group in groupby(ordered) if len(tuple(group)) > 1)


def _named_apart(registered: Sequence[_RegisteredTool]) -> tuple[_RegisteredTool, ...]:
    """Each tool keeps its gateway name where its bare name would read the same as another tool's under one label,
    as when a toolset spans two servers that each have a `search` tool."""
    shared: Final = _shared_names(each.name for each in registered)
    return tuple(
        replace(each, name=replace(each.name, name=each.tool.name)) if each.name in shared else each
        for each in registered
    )


def _listed_tool(tool: MCPTool, name: str) -> Mapping[str, object]:
    return {
        "name": name,
        "description": tool.description,
        "input_schema": tool.input_schema,
        "annotations": {"read_only": bool(tool.annotations and tool.annotations.read_only_hint)},
    }


def _list_item(server_label: str, tools: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    return {
        "id": f"mcpl_{uuid.uuid4().hex}",
        "type": "mcp_list_tools",
        "server_label": server_label,
        "tools": list(tools),
        "error": None,
    }


def _labels_in_order(names: Sequence[GatewayToolName]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(name.server_label for name in names))


def openai_mcp_items(
    registered_tools: Sequence[MCPTool],
    tool_server_map: Mapping[str, str],
    server_labels: Mapping[str, str],
    server_for: Callable[[str], MCPServer | None],
    direct: DirectListing,
    held: HeldReferences,
    labels: CallerLabels,
) -> OpenAIMCPItems:
    registered: Final = _named_apart(
        tuple(
            _registered_tool(tool, tool_server_map.get(tool.name, ""), labels, server_for) for tool in registered_tools
        )
    )
    direct_names: Final = MappingProxyType(
        {
            name: GatewayToolName(server_label=tool.server.server_label, name=tool.tool.name)
            for name, tool in direct.by_name.items()
        }
    )
    run_here: Final = tuple(each for each in registered if held.lets_run(each.server_keys, server_labels))
    registered_items: Final = tuple(
        _list_item(
            label, [_listed_tool(each.tool, each.name.name) for each in run_here if each.name.server_label == label]
        )
        for label in _labels_in_order([each.name for each in run_here])
    )
    direct_items: Final = tuple(
        _list_item(label, [_listed_tool(tool, tool.name) for tool in tools])
        for label, tools in direct.server_tools.items()
    )
    return OpenAIMCPItems(
        list_items=(*registered_items, *direct_items),
        names=MappingProxyType(
            {
                **{each.tool.name: each.name for each in run_here},
                **direct_names,
            }
        ),
    )


def _listing(item: object) -> tuple[object, object]:
    return _field(item, "tools"), _field(item, "error")


def without_listings_the_context_holds(items: OpenAIMCPItems, context: Sequence[object]) -> OpenAIMCPItems:
    """Leaves out a server's mcp_list_tools item when the latest one the conversation already holds for its
    server_label, in the stored turns or the input, lists the same tools, since OpenAI lists a server once while its
    list is in context. A listing that changed since, as after a revoke or a narrowed toolset, is listed again so the
    caller sees the change. The listing itself always comes from the server; the conversation's item only decides
    whether the caller needs to see it again."""
    in_context: Final = MappingProxyType(
        {_text(item, "server_label"): _listing(item) for item in context if _field(item, "type") == "mcp_list_tools"}
    )
    return replace(
        items,
        list_items=tuple(
            item
            for item in items.list_items
            if _text(item, "server_label") not in in_context
            or in_context[_text(item, "server_label")] != _listing(item)
        ),
    )


def mcp_call_item(
    item_id: str, tool: GatewayToolName, arguments: str, output: str | None, error: str | None
) -> Mapping[str, object]:
    return {
        "id": item_id,
        "type": "mcp_call",
        "status": "failed" if error else "completed",
        "approval_request_id": None,
        "arguments": arguments,
        "error": error,
        "name": tool.name,
        "output": None if error else output,
        "server_label": tool.server_label,
    }


def _gateway_name(names: Mapping[str, GatewayToolName], item: object) -> GatewayToolName:
    name: Final = _text(item, "name") or ""
    return names.get(name) or GatewayToolName(server_label=_text(item, "server_label") or "unlabeled", name=name)


def _is_gateway_function_call(names: Mapping[str, GatewayToolName], item: object) -> bool:
    return _field(item, "type") == "function_call" and _text(item, "name") in names


def _reshaped_item(names: Mapping[str, GatewayToolName], item: object) -> object:
    return _openai_mcp_call(names, item) if _field(item, "type") == "mcp_call" else item


def _openai_mcp_call(names: Mapping[str, GatewayToolName], item: object) -> Mapping[str, object]:
    return mcp_call_item(
        item_id=_text(item, "id") or f"mcp_{uuid.uuid4().hex}",
        tool=_gateway_name(names, item),
        arguments=_text(item, "arguments") or "",
        output=_text(item, "output"),
        error=_text(item, "error"),
    )


def openai_output(items: OpenAIMCPItems, output: Sequence[object]) -> tuple[object, ...]:
    """A response's output as OpenAI lists it: the mcp_list_tools items first, then every item the model produced.
    Each call the gateway ran is already an mcp_call here, so a gateway function_call left in it was handed back."""
    return (*items.list_items, *(_reshaped_item(items.names, item) for item in output))


def _response_item(item: Mapping[str, object]) -> BaseLiteLLMOpenAIResponseObject:
    return BaseLiteLLMOpenAIResponseObject(**item)


def _list_events(index: int, item: Mapping[str, object]) -> tuple[ResponsesAPIStreamingResponse, ...]:
    item_id: Final = _text(item, "id") or ""
    return (
        OutputItemAddedEvent(
            type=ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
            output_index=index,
            item=_response_item({**item, "tools": []}),
        ),
        MCPListToolsInProgressEvent(
            type=ResponsesAPIStreamEvents.MCP_LIST_TOOLS_IN_PROGRESS,
            sequence_number=0,
            output_index=index,
            item_id=item_id,
        ),
        MCPListToolsCompletedEvent(
            type=ResponsesAPIStreamEvents.MCP_LIST_TOOLS_COMPLETED,
            sequence_number=0,
            output_index=index,
            item_id=item_id,
        ),
        OutputItemDoneEvent(
            type=ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE,
            sequence_number=0,
            output_index=index,
            item=_response_item(item),
        ),
    )


def _copied(event: ResponsesAPIStreamingResponse, update: Mapping[str, object]) -> ResponsesAPIStreamingResponse:
    return event.model_copy(update=dict(update))


def _failed(held: ResponsesAPIStreamingResponse) -> ResponsesAPIStreamingResponse:
    output_index: Final = _field(held, "output_index")
    return MCPCallFailedEvent(
        type=ResponsesAPIStreamEvents.MCP_CALL_FAILED,
        sequence_number=0,
        item_id=_text(held, "item_id") or "",
        output_index=output_index if isinstance(output_index, int) else 0,
    )


@dataclass(frozen=True, slots=True)
class OpenAIStreamShaper:
    """Rewrites the gateway's MCP stream into OpenAI's event order. The mcp_list_tools items open the output, the
    function_call events of a gateway tool are withheld since its mcp_call events stand for it once it runs, and are
    released before the final event when the response hands that call back unrun. Output indexes are renumbered
    densely, and every event gets the next sequence number."""

    items: OpenAIMCPItems
    listed: bool = False
    indexes: Mapping[int, int] = field(default_factory=lambda: MappingProxyType({}))
    withheld: Mapping[int, tuple[ResponsesAPIStreamingResponse, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    held: ResponsesAPIStreamingResponse | None = None
    sequence: int = 0

    def shape(
        self, event: ResponsesAPIStreamingResponse
    ) -> tuple["OpenAIStreamShaper", tuple[ResponsesAPIStreamingResponse, ...]]:
        opening: Final = _field(event, "type") in _OPENING_EVENTS
        listing: Final = () if self.listed or opening else self._listing_events()
        listed_state: Final = self if self.listed or opening else replace(self, listed=True)
        shaped_state, shaped = listed_state._shaped(event)
        numbered: Final = tuple(
            _copied(each, {"sequence_number": shaped_state.sequence + offset})
            for offset, each in enumerate((*listing, *shaped))
        )
        return replace(shaped_state, sequence=shaped_state.sequence + len(numbered)), numbered

    def round_output(
        self, output: Sequence[object], ran_calls: Mapping[str, Mapping[str, object]]
    ) -> tuple[object, ...]:
        return ran_round_output(output, ran_calls)

    def client_output(self, output: Sequence[object]) -> tuple[object, ...]:
        return openai_output(self.items, output)

    def _listing_events(self) -> tuple[ResponsesAPIStreamingResponse, ...]:
        return tuple(chain.from_iterable(_list_events(index, item) for index, item in enumerate(self.items.list_items)))

    def _public_index(self, index: int) -> tuple["OpenAIStreamShaper", int]:
        if index in self.indexes:
            return self, self.indexes[index]
        public: Final = len(self.items.list_items) + len(self.indexes)
        return replace(self, indexes=MappingProxyType({**self.indexes, index: public})), public

    def _shaped(
        self, event: ResponsesAPIStreamingResponse
    ) -> tuple["OpenAIStreamShaper", tuple[ResponsesAPIStreamingResponse, ...]]:
        event_type: Final = _field(event, "type")
        index: Final = _field(event, "output_index")
        item: Final = _field(event, "item")
        if event_type in _TERMINAL_EVENTS:
            final_state, handed_back_events = self._released(event)
            return final_state, (*handed_back_events, self._terminal(event))
        if not isinstance(index, int):
            return self, (event,)
        if index in self.withheld:
            return self._withholding(index, (*self.withheld[index], event)), ()
        if event_type == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED and _is_gateway_function_call(
            self.items.names, item
        ):
            return self._withholding(index, (event,)), ()
        indexed_state, public = self._public_index(index)
        moved: Final = _copied(event, {"output_index": public})
        if event_type == ResponsesAPIStreamEvents.MCP_CALL_COMPLETED:
            return replace(indexed_state, held=moved), ()
        if _field(item, "type") != "mcp_call":
            return indexed_state, (moved,)
        reshaped: Final = _openai_mcp_call(self.items.names, item)
        in_progress: Final = (
            {**reshaped, "status": "in_progress", "output": None}
            if event_type == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED
            else reshaped
        )
        item_event: Final = _copied(moved, {"item": _response_item(in_progress)})
        held: Final = indexed_state.held
        released: Final = () if held is None else ((_failed(held) if _text(item, "error") else held),)
        return replace(indexed_state, held=None), (*released, item_event)

    def _withholding(self, index: int, events: tuple[ResponsesAPIStreamingResponse, ...]) -> "OpenAIStreamShaper":
        return replace(self, withheld=MappingProxyType({**self.withheld, index: events}))

    def _released(
        self, terminal: ResponsesAPIStreamingResponse
    ) -> tuple["OpenAIStreamShaper", tuple[ResponsesAPIStreamingResponse, ...]]:
        response: Final = _field(terminal, "response")
        handed_back: Final = frozenset(
            _call_id(item)
            for item in (output_items(response) if isinstance(response, ResponsesAPIResponse) else ())
            if _is_gateway_function_call(self.items.names, item)
        )
        indexes: Final = tuple(
            index for index in sorted(self.withheld) if _call_id(_field(self.withheld[index][0], "item")) in handed_back
        )
        start: Final[tuple[OpenAIStreamShaper, tuple[ResponsesAPIStreamingResponse, ...]]] = (self, ())
        return reduce(lambda released, index: released[0]._release(released[1], index), indexes, start)

    def _release(
        self, events: tuple[ResponsesAPIStreamingResponse, ...], index: int
    ) -> tuple["OpenAIStreamShaper", tuple[ResponsesAPIStreamingResponse, ...]]:
        indexed_state, public = self._public_index(index)
        return indexed_state, (*events, *(_copied(each, {"output_index": public}) for each in self.withheld[index]))

    def _terminal(self, event: ResponsesAPIStreamingResponse) -> ResponsesAPIStreamingResponse:
        response: Final = _field(event, "response")
        if not isinstance(response, ResponsesAPIResponse):
            return event
        return _copied(
            event,
            {
                "response": response.model_copy(
                    update={"output": list(openai_output(self.items, output_items(response)))}
                )
            },
        )


def _call_id(item: object) -> str | None:
    return _text(item, "call_id") or _text(item, "id")


def _is_empty_message(item: object) -> bool:
    if _field(item, "type") != "message":
        return False
    try:
        parts: Final = _MESSAGE_PARTS.validate_python(_field(item, "content"))
    except ValidationError:
        return False
    return all(_field(part, "type") == "output_text" and not _text(part, "text") for part in parts)


def _is_ran_call(item: object, ran_calls: Mapping[str, Mapping[str, object]]) -> bool:
    return _field(item, "type") == "function_call" and _call_id(item) in ran_calls


def ran_round_output(output: Sequence[object], ran_calls: Mapping[str, Mapping[str, object]]) -> tuple[object, ...]:
    """A tool round's output with each call the gateway ran, keyed by call id, moved after the round's other items as
    its mcp_call, and without the empty assistant message a bridged provider adds beside its tool calls. The mcp_call
    items come last because the gateway runs a call only once the model's round ends, so this is also the order a
    stream adds the items in."""
    return (
        *(item for item in output if not (_is_ran_call(item, ran_calls) or _is_empty_message(item))),
        *ran_calls.values(),
    )
