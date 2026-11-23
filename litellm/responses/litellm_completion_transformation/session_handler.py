import asyncio
import json
from bisect import bisect_left
from collections.abc import Container, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial, reduce
from itertools import chain, repeat
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Protocol, cast

from pydantic import TypeAdapter, ValidationError

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    LITELLM_TRUNCATED_PAYLOAD_FIELD,
    PREVIOUS_RESPONSE_IDS_METADATA_KEY,
    REDACTED_BY_LITELLM,
    REDACTED_TOOL_CALL_ARGUMENTS_PLACEHOLDER,
)
from litellm.litellm_core_utils.cold_storage_handler import ColdStorageHandler
from litellm.litellm_core_utils.core_helpers import as_str_mapping, attribute_or_none, require_attributes
from litellm.litellm_core_utils.tool_fields import (
    SentToolNames,
    declared_function_names,
    with_renamed_tool_calls,
    with_tool_calls_named_as_declared,
)
from litellm.responses.additional_tools import with_additional_tools_hoisted
from litellm.responses.litellm_completion_transformation.client_tool_search import (
    bridge_client_tool_search,
    bridged_client_tool_search_input,
    client_tool_search_loaded_tools,
    tools_loaded_by_client_tool_search,
)
from litellm.responses.litellm_completion_transformation.compaction_items import (
    holds_compaction_summary,
    input_items,
    instruction_items,
    response_holds_compaction_block,
)
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import (
    ALL_RESPONSES_API_TOOL_PARAMS,
    AllMessageValues,
    ChatCompletionResponseMessage,
    GenericChatCompletionMessage,
    ResponseInputParam,
)
from litellm.types.utils import CallTypes, ChatCompletionMessageToolCall, Message, ModelResponse

if TYPE_CHECKING:
    from litellm.proxy._types import SpendLogsMetadata
    from litellm.responses.litellm_completion_transformation.transformation import (
        ChatCompletionSession,
        SessionMessage,
    )
else:
    ChatCompletionSession = Any

########################################################
# Cold Storage Handler
########################################################
COLD_STORAGE_HANDLER: Final = ColdStorageHandler()
########################################################


class _CarriesArguments(Protocol):
    arguments: object


def _normalize_redacted_function_arguments(tool_call: object) -> None:
    function: Final = attribute_or_none(tool_call, "function")
    if function is None:
        return
    carrier: Final = require_attributes(function, _CarriesArguments, "arguments")
    if carrier.arguments == REDACTED_BY_LITELLM:
        carrier.arguments = REDACTED_TOOL_CALL_ARGUMENTS_PLACEHOLDER


def _normalize_redacted_tool_call_arguments(message: Message) -> None:
    """Redaction stores the bare sentinel (invalid JSON) in tool-call arguments;
    normalize replayed history to "{}" so provider converters can parse it."""
    for tool_call in message.tool_calls or []:
        _normalize_redacted_function_arguments(tool_call)
    function_call: Final = message.function_call
    if function_call is not None and function_call.arguments == REDACTED_BY_LITELLM:
        function_call.arguments = REDACTED_TOOL_CALL_ARGUMENTS_PLACEHOLDER


_MCP_TOOL_CALL_TYPES: Final = frozenset((CallTypes.list_mcp_tools.value, CallTypes.call_mcp_tool.value))


_FAILED_TURN_STATUS: Final = "failure"


def _is_a_failed_turn(spend_log: Mapping[str, object]) -> bool:
    return "status" in spend_log and spend_log["status"] == _FAILED_TURN_STATUS


def _is_a_conversation_turn(spend_log: Mapping[str, object]) -> bool:
    return spend_log.get("call_type") not in _MCP_TOOL_CALL_TYPES and not _is_a_failed_turn(spend_log)


_RESPONSE_READ_CALL_TYPES: Final = frozenset(("aget_responses", "acancel_responses"))


def _output_item_id(item: object) -> str | None:
    match item:
        case {"id": str() as item_id}:
            return item_id
        case _:
            return None


def _is_a_response_read(spend_log: Mapping[str, object]) -> bool:
    return "call_type" in spend_log and spend_log["call_type"] in _RESPONSE_READ_CALL_TYPES


def _stored_response(spend_log: Mapping[str, object]) -> object:
    return spend_log.get("response")


def _stored_model_map_key(metadata: object) -> str | None:
    match json.loads(metadata) if isinstance(metadata, str) and metadata else metadata:
        case {"model_map_information": {"model_map_key": str() as model_map_key}}:
            return model_map_key
        case _:
            return None


def _stored_turn_fills_empty_user_turns(spend_log: Mapping[str, object]) -> bool:
    from litellm.responses.litellm_completion_transformation.transformation import target_fills_empty_user_turns

    fields: Final = as_str_mapping(spend_log) or {}
    model: Final = fields.get("model")
    provider: Final = fields.get("custom_llm_provider")
    return target_fills_empty_user_turns(
        model if isinstance(model, str) else "",
        provider if isinstance(provider, str) else None,
        _stored_model_map_key(fields.get("metadata")),
    )


def _output_items(spend_log: Mapping[str, object]) -> Iterator[tuple[str | None, object]]:
    output: Final = _stored_responses_api_output(_stored_response(spend_log)) or []
    return zip(map(_output_item_id, output), output)


def _identified_output_items(spend_log: Mapping[str, object]) -> Iterator[tuple[str, object]]:
    return ((item_id, item) for item_id, item in _output_items(spend_log) if item_id is not None)


def _output_item_ids(spend_log: Mapping[str, object]) -> frozenset[str]:
    return frozenset(item_id for item_id, _ in _identified_output_items(spend_log))


def _first_index_holding_each_item(indexed_turns: Iterable[tuple[int, Mapping[str, object]]]) -> Mapping[str, int]:
    return dict(
        chain.from_iterable(
            ((item_id, index) for item_id in _output_item_ids(turn)) for index, turn in reversed(tuple(indexed_turns))
        )
    )


def _turn_replaying_each_item(turns: Sequence[Mapping[str, object]]) -> Mapping[str, int]:
    indexed_turns: Final = tuple(enumerate(turns))
    first_read_holding: Final = _first_index_holding_each_item(
        (index, turn) for index, turn in indexed_turns if _is_a_response_read(turn)
    )
    first_other_turn_holding: Final = _first_index_holding_each_item(
        (index, turn) for index, turn in indexed_turns if not _is_a_response_read(turn)
    )
    return MappingProxyType({**first_read_holding, **first_other_turn_holding})


def _latest_read_copy_of_each_item(turns: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    return MappingProxyType(
        dict(chain.from_iterable(_identified_output_items(turn) for turn in turns if _is_a_response_read(turn)))
    )


def _with_only_new_output(
    index: int, turn: Mapping[str, object], turn_replaying: Mapping[str, int], latest_copy: Mapping[str, object]
) -> Mapping[str, object] | None:
    if not _is_a_response_read(turn):
        return turn
    stored_response: Final = as_str_mapping(_stored_response(turn))
    new_output: Final = [
        item if item_id is None else latest_copy[item_id]
        for item_id, item in _output_items(turn)
        if item_id is None or turn_replaying.get(item_id) == index
    ]
    if stored_response is None or not new_output:
        return None
    replayed_turn: Final[Mapping[str, object]] = {**turn, "response": {**stored_response, "output": new_output}}
    return replayed_turn


def _conversation_turns(spend_logs: Sequence[Mapping[str, object]]) -> tuple[Mapping[str, object], ...]:
    turns: Final = tuple(filter(_is_a_conversation_turn, spend_logs))
    turn_replaying: Final = _turn_replaying_each_item(turns)
    latest_copy: Final = _latest_read_copy_of_each_item(turns)
    replayed: Final = (
        _with_only_new_output(index, turn, turn_replaying, latest_copy) for index, turn in enumerate(turns)
    )
    return tuple(turn for turn in replayed if turn is not None)


_MCP_AUTO_EXECUTE_OUTPUT_TYPES: Final = frozenset(("mcp_tools_fetched", "tool_execution_results"))


def _is_model_output(item: object) -> bool:
    return not (isinstance(item, Mapping) and item.get("type") in _MCP_AUTO_EXECUTE_OUTPUT_TYPES)


def _stored_responses_api_output(stored_response: object) -> ResponseInputParam | None:
    match stored_response:
        case {"object": "response", "output": list() as output}:
            return [item for item in output if _is_model_output(item)]
        case {"object": "response"}:
            return []
        case _:
            return None


@dataclass(frozen=True, slots=True)
class _ReplayStart:
    index: int
    at_output: bool


@dataclass(frozen=True, slots=True)
class StoredTurns:
    spend_logs: tuple[Mapping[str, object], ...]
    proxy_server_requests: tuple[Mapping[str, object] | None, ...]


def _compaction_point(
    index: int, spend_log: Mapping[str, object], proxy_server_request: Mapping[str, object] | None
) -> _ReplayStart | None:
    if response_holds_compaction_block(_stored_response(spend_log)):
        return _ReplayStart(index=index, at_output=True)
    if proxy_server_request is not None and holds_compaction_summary(proxy_server_request.get("input")):
        return _ReplayStart(index=index, at_output=False)
    return None


def _replay_start(
    spend_logs: Sequence[Mapping[str, object]], proxy_server_requests: Sequence[Mapping[str, object] | None]
) -> _ReplayStart:
    return next(
        (
            start
            for index in range(len(spend_logs) - 1, -1, -1)
            if (start := _compaction_point(index, spend_logs[index], proxy_server_requests[index])) is not None
        ),
        _ReplayStart(index=0, at_output=False),
    )


def _holds_compaction_point(
    spend_logs: Sequence[Mapping[str, object]], proxy_server_requests: Sequence[Mapping[str, object] | None]
) -> bool:
    return any(
        _compaction_point(index, spend_log, proxy_server_request) is not None
        for index, (spend_log, proxy_server_request) in enumerate(zip(spend_logs, proxy_server_requests, strict=True))
    )


def _referenced_request_ids(previous_response_id: str, client_response_ids: tuple[str, ...]) -> tuple[str, ...]:
    decoded_response_id: Final = ResponsesAPIRequestUtils.decode_responses_api_response_id(previous_response_id)
    response_id: Final = decoded_response_id.get("response_id", previous_response_id)
    return tuple(dict.fromkeys((response_id, previous_response_id, *client_response_ids)))


def _stored_previous_response_ids(spend_log: Mapping[str, object]) -> tuple[str, ...] | None:
    """The path to the root a turn stored on its row, None for a row written before paths were stored."""
    metadata: Final = spend_log.get("metadata")
    try:
        parsed: Final = (
            _STORED_METADATA_ADAPTER.validate_json(metadata)
            if isinstance(metadata, str)
            else _STORED_METADATA_ADAPTER.validate_python(metadata)
        )
    except ValidationError:
        return None
    try:
        return tuple(_REQUEST_IDS_ADAPTER.validate_python(parsed.get(PREVIOUS_RESPONSE_IDS_METADATA_KEY)))
    except ValidationError:
        return None


def _path_through_referenced_rows(
    referenced_rows: Sequence[Mapping[str, object]], request_ids: Sequence[str]
) -> tuple[str, ...] | None:
    """The request ids to replay, root first, ending at the referenced response, or None without a stored path."""
    stored_path: Final = next(
        (path for path in map(_stored_previous_response_ids, referenced_rows) if path is not None), None
    )
    if stored_path is None:
        return None
    return tuple(dict.fromkeys((*stored_path, *request_ids)))


def _in_path_order(spend_logs: Sequence[Mapping[str, object]], path: Sequence[str]) -> Sequence[Mapping[str, object]]:
    position: Final[Mapping[object, int]] = {request_id: index for index, request_id in enumerate(path)}
    return sorted(spend_logs, key=lambda spend_log: position.get(spend_log.get("request_id"), len(path)))


def _turns_through_referenced_response(
    spend_logs: Sequence[Mapping[str, object]], referenced_request_ids: tuple[str, ...]
) -> int:
    return next(
        (
            index + 1
            for index in range(len(spend_logs) - 1, -1, -1)
            if spend_logs[index].get("request_id") in referenced_request_ids
        ),
        len(spend_logs),
    )


def _replayed_request(
    index: int,
    proxy_server_request: Mapping[str, object] | None,
    start: _ReplayStart,
    earlier_instructions: tuple[object, ...],
) -> Mapping[str, object] | None:
    if proxy_server_request is None or index != start.index:
        return proxy_server_request
    stored_input: Final = proxy_server_request.get("input")
    if start.at_output:
        return {
            **proxy_server_request,
            "input": [*earlier_instructions, *instruction_items(stored_input)],
            "messages": None,
        }
    if not earlier_instructions:
        return proxy_server_request
    return {**proxy_server_request, "input": [*earlier_instructions, *input_items(stored_input)]}


def _instruction_items_before(
    proxy_server_requests: Sequence[Mapping[str, object] | None], start: _ReplayStart
) -> tuple[object, ...]:
    return tuple(
        chain.from_iterable(
            instruction_items(request.get("input"))
            for request in proxy_server_requests[: start.index]
            if request is not None
        )
    )


_NO_TOOL_NAMES: Final = SentToolNames(member_chat_names=MappingProxyType({}), self_named_namespace_members=frozenset())


@dataclass(frozen=True, slots=True)
class _FirstLoads:
    turn_by_name: Mapping[str, int]
    sorted_turns: tuple[int, ...]

    def loaded_before(self, name: str, turn: int) -> bool:
        first_turn: Final = self.turn_by_name.get(name)
        return first_turn is not None and first_turn < turn

    def count_before(self, turn: int) -> int:
        return bisect_left(self.sorted_turns, turn)


def _first_loads(names_by_turn: Iterable[Iterable[str]]) -> _FirstLoads:
    turn_by_name: Final = MappingProxyType(
        dict(reversed(tuple(chain.from_iterable(zip(names, repeat(turn)) for turn, names in enumerate(names_by_turn)))))
    )
    return _FirstLoads(turn_by_name=turn_by_name, sorted_turns=tuple(sorted(turn_by_name.values())))


@dataclass(frozen=True, slots=True)
class _SessionLoads:
    chat_names: Mapping[str, str]
    chat_name_loads: _FirstLoads
    self_named_loads: _FirstLoads


@dataclass(frozen=True, slots=True, eq=False)
class _ChatNamesSentWith(Mapping[str, str]):
    own: Mapping[str, str]
    loads: _SessionLoads
    turn: int

    def _loaded_before(self, name: str) -> bool:
        return self.loads.chat_name_loads.loaded_before(name, self.turn)

    def __getitem__(self, name: str) -> str:
        if name in self.own:
            return self.own[name]
        if self._loaded_before(name):
            return self.loads.chat_names[name]
        raise KeyError(name)

    def __iter__(self) -> Iterator[str]:
        return chain(
            self.own, (name for name in self.loads.chat_names if name not in self.own and self._loaded_before(name))
        )

    def __len__(self) -> int:
        loaded_before_turn: Final = self.loads.chat_name_loads.count_before(self.turn)
        return len(self.own) + loaded_before_turn - sum(1 for name in self.own if self._loaded_before(name))


@dataclass(frozen=True, slots=True)
class _SelfNamedSentWith(Container[str]):
    own: Container[str]
    loads: _FirstLoads
    turn: int

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and (name in self.own or self.loads.loaded_before(name, self.turn))


def _tool_names(tools: object) -> SentToolNames:
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    return LiteLLMCompletionResponsesConfig.sent_tool_names(
        LiteLLMCompletionResponsesConfig.request_tools({"tools": tools})
    )


def _tools_loaded_in(proxy_server_request: Mapping[str, object] | None) -> tuple[ALL_RESPONSES_API_TOOL_PARAMS, ...]:
    return () if proxy_server_request is None else tools_loaded_by_client_tool_search(proxy_server_request.get("input"))


def _session_loads(proxy_server_requests: Sequence[Mapping[str, object] | None]) -> _SessionLoads:
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    loaded_tools: Final = tuple(
        LiteLLMCompletionResponsesConfig.request_tools({"tools": list(_tools_loaded_in(request))})
        for request in proxy_server_requests
    )
    chat_names_by_turn: Final = tuple(map(LiteLLMCompletionResponsesConfig.member_chat_names, loaded_tools))
    return _SessionLoads(
        chat_names=MappingProxyType(dict(chain.from_iterable(names.items() for names in chat_names_by_turn))),
        chat_name_loads=_first_loads(chat_names_by_turn),
        self_named_loads=_first_loads(map(LiteLLMCompletionResponsesConfig.self_named_namespace_members, loaded_tools)),
    )


def _tool_names_sent_with(
    proxy_server_request: Mapping[str, object] | None, loads: _SessionLoads, turn: int
) -> SentToolNames:
    """The tool names a stored turn was sent with, so its calls replay under the names the provider saw."""
    if proxy_server_request is None:
        return _NO_TOOL_NAMES
    hoisted: Final = with_additional_tools_hoisted(proxy_server_request, drop_params=False)
    request_input: Final = hoisted.get("input")
    tools: Final = hoisted.get("tools")
    if not isinstance(request_input, (str, list)) or not (tools is None or isinstance(tools, list)):
        return _tool_names(tools)
    searched: Final = bridge_client_tool_search(
        cast(ResponseInputParam, request_input),  # cast-ok: forwarded as sent, as inference does
        cast("list[ALL_RESPONSES_API_TOOL_PARAMS] | None", tools),  # cast-ok: forwarded as sent
    )
    own: Final = _tool_names(list(searched.tools))
    if not searched.bridged:
        return own
    return SentToolNames(
        member_chat_names=_ChatNamesSentWith(own.member_chat_names, loads, turn),
        self_named_namespace_members=_SelfNamedSentWith(own.self_named_namespace_members, loads.self_named_loads, turn),
    )


def _tool_names_by_turn(proxy_server_requests: Sequence[Mapping[str, object] | None]) -> tuple[SentToolNames, ...]:
    loads: Final = _session_loads(proxy_server_requests)
    return tuple(_tool_names_sent_with(request, loads, turn) for turn, request in enumerate(proxy_server_requests))


def _rebuilt_from_cold_storage(spend_log: Mapping[str, object], proxy_server_request: Mapping[str, object]) -> bool:
    return "input" not in proxy_server_request and LITELLM_TRUNCATED_PAYLOAD_FIELD in str(
        spend_log.get("proxy_server_request")
    )


def _declared_tool_names(
    spend_log: Mapping[str, object], proxy_server_request: Mapping[str, object] | None
) -> frozenset[str]:
    if proxy_server_request is None or not _rebuilt_from_cold_storage(spend_log, proxy_server_request):
        return frozenset()
    model_parameters: Final = as_str_mapping(proxy_server_request.get("model_parameters"))
    return frozenset() if model_parameters is None else declared_function_names(model_parameters.get("tools"))


def _stored_instructions(proxy_server_request: Mapping[str, object] | None) -> str | None:
    instructions: Final = None if proxy_server_request is None else proxy_server_request.get("instructions")
    return instructions if isinstance(instructions, str) and instructions else None


def _latest_replayable_instructions(proxy_server_requests: Sequence[Mapping[str, object] | None]) -> str | None:
    latest: Final = next(
        (instructions for instructions in map(_stored_instructions, reversed(proxy_server_requests)) if instructions),
        None,
    )
    return None if latest == REDACTED_BY_LITELLM else latest


_STORED_METADATA_ADAPTER: Final = TypeAdapter(dict[str, object])
_REQUEST_IDS_ADAPTER: Final = TypeAdapter(list[str])

_REFERENCED_ROWS_QUERY: Final = """
    SELECT *
    FROM "LiteLLM_SpendLogs"
    WHERE request_id = ANY($1::text[]);
"""

_SAME_OWNER_AS_REFERENCED: Final = """
    EXISTS (
        SELECT 1
        FROM matching_session
        WHERE {same_conversation}
          AND (
              matching_session."user" = logs."user"
              OR (matching_session."user" IS NULL AND logs."user" IS NULL)
          )
          AND (
              matching_session.team_id = logs.team_id
              OR (matching_session.team_id IS NULL AND logs.team_id IS NULL)
          )
          AND (
              COALESCE(matching_session."user", '') <> ''
              OR COALESCE(matching_session.team_id, '') <> ''
              OR (
                  COALESCE(matching_session.api_key, '') <> ''
                  AND matching_session.api_key = logs.api_key
              )
          )
    )
"""

_OWNER_SCOPED_QUERY: Final = """
    WITH matching_session AS (
        SELECT session_id, api_key, "user", team_id
        FROM "LiteLLM_SpendLogs"
        WHERE request_id = ANY($1::text[])
    )
    SELECT *
    FROM "LiteLLM_SpendLogs" AS logs
    WHERE {same_owner}
    ORDER BY "endTime" ASC;
"""

_SESSION_QUERY: Final = _OWNER_SCOPED_QUERY.format(
    same_owner=_SAME_OWNER_AS_REFERENCED.format(same_conversation="matching_session.session_id = logs.session_id")
)

_PATH_QUERY: Final = _OWNER_SCOPED_QUERY.format(
    same_owner=_SAME_OWNER_AS_REFERENCED.format(same_conversation="logs.request_id = ANY($2::text[])")
)


class ResponsesSessionHandler:
    @staticmethod
    async def get_chat_completion_message_history_for_previous_response_id(
        previous_response_id: str,
        item_cache_control: bool = False,
        client_response_ids: tuple[str, ...] = (),
        tool_call_provider_fields: bool = False,
        stored_turns: StoredTurns | None = None,
        empty_user_turn_text: str | None = None,
        keep_tool_references: bool = False,
    ) -> ChatCompletionSession:
        """
        Return the chat completion message history for a previous response id
        """
        from litellm.responses.litellm_completion_transformation.transformation import (
            ChatCompletionSession,
            with_stored_calls_the_input_answers,
        )

        verbose_proxy_logger.debug("inside get_chat_completion_message_history_for_previous_response_id")
        turns: Final = (
            stored_turns
            if stored_turns is not None
            else await ResponsesSessionHandler.get_stored_turns_for_previous_response_id(
                previous_response_id, client_response_ids
            )
        )
        all_spend_logs: Final = turns.spend_logs
        verbose_proxy_logger.debug("found %s spend logs for this response id", len(all_spend_logs))

        litellm_session_id: object = None
        if len(all_spend_logs) > 0:
            litellm_session_id = all_spend_logs[0].get("session_id")

        proxy_server_requests: Final = turns.proxy_server_requests
        replayable_turns: Final = _turns_through_referenced_response(
            all_spend_logs, _referenced_request_ids(previous_response_id, client_response_ids)
        )
        start: Final = _replay_start(all_spend_logs[:replayable_turns], proxy_server_requests[:replayable_turns])
        replayed_turns: Final = (
            replayable_turns if _holds_compaction_point(all_spend_logs, proxy_server_requests) else len(all_spend_logs)
        )
        earlier_instructions: Final = _instruction_items_before(proxy_server_requests, start)
        names_by_turn: Final = _tool_names_by_turn(proxy_server_requests)
        replayed_history: Final = ResponsesSessionHandler._history_of_turns(
            all_spend_logs[start.index : replayed_turns],
            [
                _replayed_request(index, proxy_server_requests[index], start, earlier_instructions)
                for index in range(start.index, replayed_turns)
            ],
            names_by_turn[start.index : replayed_turns],
            item_cache_control,
            tool_call_provider_fields,
            empty_user_turn_text,
            keep_tool_references,
        )
        history_before_replay: Final = ResponsesSessionHandler._history_of_turns(
            all_spend_logs[: start.index],
            proxy_server_requests[: start.index],
            names_by_turn[: start.index],
            item_cache_control,
            tool_call_provider_fields,
            empty_user_turn_text,
            keep_tool_references,
        )
        chat_completion_message_history: Final = (
            list(with_stored_calls_the_input_answers(replayed_history, history_before_replay))
            if history_before_replay
            else replayed_history
        )

        verbose_proxy_logger.debug(
            "chat_completion_message_history %s",
            json.dumps(chat_completion_message_history, indent=4, default=str),
        )
        return ChatCompletionSession(
            messages=chat_completion_message_history,
            litellm_session_id=litellm_session_id,
            instructions=_latest_replayable_instructions(proxy_server_requests),
            previous_response_ids=tuple(spend_log["request_id"] for spend_log in all_spend_logs[:replayed_turns]),
        )

    @staticmethod
    async def get_stored_turns_for_previous_response_id(
        previous_response_id: str,
        client_response_ids: tuple[str, ...] = (),
    ) -> StoredTurns:
        spend_logs: Final = _conversation_turns(
            await ResponsesSessionHandler.get_all_spend_logs_for_previous_response_id(
                previous_response_id, client_response_ids
            )
        )
        return StoredTurns(
            spend_logs=spend_logs,
            proxy_server_requests=tuple(
                [
                    await ResponsesSessionHandler.get_proxy_server_request_from_spend_log(spend_log=spend_log)
                    for spend_log in spend_logs
                ]
            ),
        )

    @staticmethod
    def client_tool_search_tools_loaded_before(
        stored_turns: StoredTurns,
        previous_response_id: str,
        client_response_ids: tuple[str, ...] = (),
    ) -> tuple[Mapping[str, object], ...]:
        turns: Final = _turns_through_referenced_response(
            stored_turns.spend_logs, _referenced_request_ids(previous_response_id, client_response_ids)
        )
        return tuple(
            chain.from_iterable(
                client_tool_search_loaded_tools(request.get("input"))
                for request in stored_turns.proxy_server_requests[:turns]
                if request is not None
            )
        )

    @staticmethod
    def _history_of_turns(
        spend_logs: Sequence[Mapping[str, object]],
        proxy_server_requests: Sequence[Mapping[str, object] | None],
        sent_tool_names: Sequence[SentToolNames],
        item_cache_control: bool,
        tool_call_provider_fields: bool,
        empty_user_turn_text: str | None = None,
        keep_tool_references: bool = False,
    ) -> list["SessionMessage"]:
        return reduce(
            lambda history, turn: ResponsesSessionHandler.extend_chat_completion_message_with_spend_log_payload(
                spend_log=turn[0],
                proxy_server_request_dict=turn[1],
                chat_completion_message_history=history,
                item_cache_control=item_cache_control,
                tool_call_provider_fields=tool_call_provider_fields,
                empty_user_turn_text=empty_user_turn_text,
                tool_names=turn[2],
                keep_tool_references=keep_tool_references,
            ),
            zip(spend_logs, proxy_server_requests, sent_tool_names, strict=True),
            [],
        )

    @staticmethod
    def extend_chat_completion_message_with_spend_log_payload(
        spend_log: Mapping[str, object],
        proxy_server_request_dict: Mapping[str, object] | None,
        chat_completion_message_history: list[
            AllMessageValues
            | GenericChatCompletionMessage
            | ChatCompletionMessageToolCall
            | ChatCompletionResponseMessage
            | Message
        ],
        item_cache_control: bool = False,
        tool_call_provider_fields: bool = False,
        empty_user_turn_text: str | None = None,
        keep_tool_references: bool = False,
        *,
        tool_names: SentToolNames = _NO_TOOL_NAMES,
    ):
        """
        Extend the chat completion message history with the spend log payload
        """
        from litellm.responses.litellm_completion_transformation.transformation import (
            LiteLLMCompletionResponsesConfig,
            restored_empty_user_turn,
            without_blank_user_tail,
        )

        response_input_param: str | ResponseInputParam | None = None

        ############################################################
        # Add Input messages for this Spend Log
        ############################################################
        if proxy_server_request_dict:
            stored_request_input: Final = proxy_server_request_dict.get("input")
            _response_input_param: Final = (
                proxy_server_request_dict.get("messages") if stored_request_input is None else stored_request_input
            )
            if isinstance(_response_input_param, (str, list)):
                response_input_param = _response_input_param
            elif isinstance(_response_input_param, dict):
                response_input_param = cast(
                    ResponseInputParam,
                    [_response_input_param],
                )

        output_messages: Final = ResponsesSessionHandler._stored_output_messages(
            _stored_response(spend_log),
            tool_names,
            _declared_tool_names(spend_log, proxy_server_request_dict),
            item_cache_control,
            tool_call_provider_fields,
        )
        if response_input_param is not None:
            stored_input: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
                input=bridged_client_tool_search_input(response_input_param),
                responses_api_request={},
                replay_reasoning=True,
                item_cache_control=item_cache_control,
                tool_call_provider_fields=tool_call_provider_fields,
                only_since_compaction=True,
                replay_tool_search=True,
                tool_names=tool_names,
                keep_tool_references=keep_tool_references,
            )
            input_before_blank_tail: Final = without_blank_user_tail(stored_input)
            inference_continues_the_blank_tail: Final = empty_user_turn_text == "" and len(
                input_before_blank_tail
            ) < len(stored_input)
            restored_turn: Final = (
                restored_empty_user_turn(
                    response_input_param,
                    (*chat_completion_message_history, *input_before_blank_tail),
                    item_cache_control,
                    partial(_stored_turn_fills_empty_user_turns, spend_log),
                    text=empty_user_turn_text,
                )
                if empty_user_turn_text is not None and output_messages and not inference_continues_the_blank_tail
                else ()
            )
            chat_completion_message_history.extend(input_before_blank_tail if restored_turn else stored_input)
            chat_completion_message_history.extend(restored_turn)

        chat_completion_message_history.extend(output_messages)
        return chat_completion_message_history

    @staticmethod
    def _stored_output_messages(
        stored_response: object,
        tool_names: SentToolNames,
        declared_tool_names: frozenset[str],
        item_cache_control: bool,
        tool_call_provider_fields: bool,
    ) -> Sequence[
        AllMessageValues
        | GenericChatCompletionMessage
        | ChatCompletionMessageToolCall
        | ChatCompletionResponseMessage
        | Message
    ]:
        from litellm.responses.litellm_completion_transformation.transformation import (
            LiteLLMCompletionResponsesConfig,
        )

        if (responses_api_output := _stored_responses_api_output(stored_response)) is not None:
            return LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
                input=bridged_client_tool_search_input(responses_api_output),
                responses_api_request={},
                replay_reasoning=True,
                item_cache_control=item_cache_control,
                tool_call_provider_fields=tool_call_provider_fields,
                only_since_compaction=True,
                replay_tool_search=True,
                tool_names=tool_names,
            )
        if not isinstance(stored_response, dict) or not stored_response:
            return ()
        model_response: Final = with_tool_calls_named_as_declared(
            with_renamed_tool_calls(ModelResponse(**stored_response), tool_names.member_chat_names),
            declared_tool_names,
        )
        chat_completion_messages: Final = tuple(
            choice.message for choice in model_response.choices if hasattr(choice, "message")
        )
        for message in chat_completion_messages:
            _normalize_redacted_tool_call_arguments(message)
        return chat_completion_messages

    @staticmethod
    async def get_proxy_server_request_from_spend_log(
        spend_log: Mapping[str, object],
    ) -> dict[str, object] | None:
        """
        Get the parsed proxy server request from the spend log
        """
        proxy_server_request: Final = spend_log.get("proxy_server_request") or "{}"
        stored_request_mapping: Final = as_str_mapping(proxy_server_request)
        proxy_server_request_dict: dict[str, object] | None = None
        if isinstance(stored_request_mapping, dict):
            proxy_server_request_dict = stored_request_mapping
        else:
            proxy_server_request_dict = json.loads(
                proxy_server_request  # pyright: ignore[reportArgumentType]  # loads raises its own TypeError
            )

        ############################################################
        # Check if user has setup cold storage for session handling
        ############################################################
        if ResponsesSessionHandler._should_check_cold_storage_for_full_payload(proxy_server_request_dict):
            # Try to get cold storage object key from spend log metadata
            _proxy_server_request_dict: dict | None = None
            cold_storage_object_key = ResponsesSessionHandler._get_cold_storage_object_key_from_spend_log(spend_log)
            if cold_storage_object_key:
                # Use the object key directly from metadata
                _proxy_server_request_dict = (
                    await ResponsesSessionHandler.get_proxy_server_request_from_cold_storage_with_object_key(
                        object_key=cold_storage_object_key,
                    )
                )
            if _proxy_server_request_dict:
                proxy_server_request_dict = _proxy_server_request_dict

        return proxy_server_request_dict

    @staticmethod
    def _get_cold_storage_object_key_from_spend_log(
        spend_log: Mapping[str, object],
    ) -> str | None:
        """
        Extract the cold storage object key from spend log metadata.

        Args:
            spend_log: The spend log payload containing metadata

        Returns:
            Optional[str]: The cold storage object key if found, None otherwise
        """
        try:
            metadata_str: Final = spend_log.get("metadata", "{}")
            if isinstance(metadata_str, str):
                metadata_dict: Final[SpendLogsMetadata] = json.loads(metadata_str)
                return metadata_dict.get("cold_storage_object_key")
            elif isinstance(metadata_str, dict):
                return metadata_str.get("cold_storage_object_key")
            return None
        except (json.JSONDecodeError, TypeError, AttributeError):
            verbose_proxy_logger.debug("Failed to parse metadata from spend log to extract cold storage object key")
            return None

    @staticmethod
    async def get_proxy_server_request_from_cold_storage_with_object_key(
        object_key: str,
    ) -> dict[str, object] | None:
        """
        Get the proxy server request from cold storage using the object key directly.

        Args:
            object_key: The S3/GCS object key to retrieve

        Returns:
            Optional[dict]: The proxy server request dict or None if not found
        """
        verbose_proxy_logger.debug("inside get_proxy_server_request_from_cold_storage_with_object_key...")

        proxy_server_request_dict: Final = (
            await COLD_STORAGE_HANDLER.get_proxy_server_request_from_cold_storage_with_object_key(
                object_key=object_key,
            )
        )

        return proxy_server_request_dict

    @staticmethod
    def _should_check_cold_storage_for_full_payload(
        proxy_server_request_dict: Mapping[str, object] | None,
    ) -> bool:
        """
        Only check cold storage when both are true
        1. `LITELLM_TRUNCATED_PAYLOAD_FIELD` is in the proxy server request dict
        2. `litellm.cold_storage_custom_logger` is not None
        """
        configured_cold_storage_custom_logger: Final = litellm.cold_storage_custom_logger
        if configured_cold_storage_custom_logger is None:
            return False
        if proxy_server_request_dict is None:
            return True
        if len(proxy_server_request_dict) == 0:
            return True
        if LITELLM_TRUNCATED_PAYLOAD_FIELD in str(proxy_server_request_dict):
            return True
        return False

    @staticmethod
    async def get_all_spend_logs_for_previous_response_id(
        previous_response_id: str,
        client_response_ids: tuple[str, ...] = (),
    ) -> Sequence[Mapping[str, object]]:
        """
        Get all spend logs for a previous response id

        The referenced response's row is read first. A row written with a path to the root
        (``litellm_previous_response_ids`` in its metadata) names every earlier turn it continues, so
        only those rows plus the referenced one are read, which keeps a fork from replaying its
        siblings and the turns sent after the referenced response. A row written before paths were
        stored falls back to its whole session:

        SELECT session_id, api_key, "user", team_id FROM spend_logs WHERE request_id = previous_response_id, then
        SELECT * FROM spend_logs with that session_id and the same owner

        Only rows with the referenced response's owner are returned: the same user and team, and for a key with
        neither, the same key. The session id can be caller supplied, so matching on it alone would mix owners. A
        user's or team's regenerated or second key keeps the history; a key with neither does not, and turns written
        by another owner (an admin continuing the chain) are not replayed.

        A just-finished turn gets a short second chance: the worker that served it may
        still be writing its spend log when the follow-up arrives, and an empty result
        drops the whole conversation instead of erroring. Deployments that write no spend
        logs at all have nothing to wait for, so they keep the single original query.
        """
        from litellm.constants import (
            RESPONSES_SESSION_LOOKUP_MAX_ATTEMPTS,
            RESPONSES_SESSION_LOOKUP_RETRY_INTERVAL,
        )
        from litellm.proxy.proxy_server import disable_spend_logs, prisma_client
        from litellm.proxy.utils import query_raw_rows

        verbose_proxy_logger.debug("decoding response id=%s", previous_response_id)

        request_ids: Final = list(_referenced_request_ids(previous_response_id, client_response_ids))
        response_id: Final = request_ids[0]
        if prisma_client is None:
            return []

        max_attempts: Final = 1 if disable_spend_logs else RESPONSES_SESSION_LOOKUP_MAX_ATTEMPTS
        for attempt in range(max_attempts):
            if attempt:
                await asyncio.sleep(RESPONSES_SESSION_LOOKUP_RETRY_INTERVAL)
            read_spend_logs = partial(query_raw_rows, prisma_client.db)
            referenced_rows = await read_spend_logs(_REFERENCED_ROWS_QUERY, request_ids)
            if not referenced_rows:
                continue
            path = _path_through_referenced_rows(referenced_rows, request_ids)
            spend_logs = (
                _in_path_order(await read_spend_logs(_PATH_QUERY, request_ids, list(path)), path)
                if path is not None
                else await read_spend_logs(_SESSION_QUERY, request_ids)
            )
            verbose_proxy_logger.debug(
                "Found the following spend logs for previous response id %s: %s",
                response_id,
                json.dumps(spend_logs, indent=4, default=str),
            )
            return spend_logs

        verbose_proxy_logger.debug("Found no spend logs for previous response id %s", response_id)
        return []
