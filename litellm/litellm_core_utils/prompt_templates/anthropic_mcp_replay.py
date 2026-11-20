from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import accumulate, chain
from types import MappingProxyType
from typing import Final

from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.prompt_templates.mid_conversation_system import message_field, parts_of

_NO_IDS: Final = frozenset[str]()
_NO_RESULTS: Final[tuple[Mapping[str, object], ...]] = ()
_MCP_BLOCK_TYPES: Final = frozenset({"mcp_tool_use", "mcp_tool_result"})


def mcp_server_name(tool_call: object) -> str | None:
    server_name: Final = message_field(
        message_field(message_field(tool_call, "function"), "provider_specific_fields"), "server_name"
    )
    return server_name if isinstance(server_name, str) and server_name else None


@dataclass(frozen=True, slots=True)
class _AssistantTurn:
    message_index: int
    run: int
    use_ids: tuple[str, ...]
    results: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class McpReplayPlan:
    dropped_ids_by_message: Mapping[int, frozenset[str]]
    lone_results_by_message: Mapping[int, tuple[Mapping[str, object], ...]]
    answered_later_ids: frozenset[str]

    def dropped_ids_at(self, message_index: int) -> frozenset[str]:
        return self.dropped_ids_by_message.get(message_index, _NO_IDS)

    def lone_results_at(self, message_index: int) -> tuple[Mapping[str, object], ...]:
        return self.lone_results_by_message.get(message_index, _NO_RESULTS)


_NOTHING_TO_CHANGE: Final = McpReplayPlan(MappingProxyType({}), MappingProxyType({}), _NO_IDS)


def _mcp_use_ids(message: object) -> tuple[str, ...]:
    tool_calls: Final = chain(
        parts_of(message_field(message, "tool_calls")),
        parts_of(message_field(message_field(message, "provider_specific_fields"), "server_tool_calls")),
    )
    return tuple(
        dict.fromkeys(
            call_id
            for call in tool_calls
            if mcp_server_name(call) is not None and isinstance(call_id := message_field(call, "id"), str)
        )
    )


def _mcp_tool_result_pair(block: object) -> tuple[str, Mapping[str, object]] | None:
    result: Final = as_str_mapping(block)
    if result is None or result.get("type") != "mcp_tool_result":
        return None
    tool_use_id: Final = result.get("tool_use_id")
    return (tool_use_id, result) if isinstance(tool_use_id, str) else None


def _mcp_tool_results(message: object) -> Mapping[str, Mapping[str, object]]:
    blocks: Final = parts_of(message_field(message_field(message, "provider_specific_fields"), "tool_results"))
    return MappingProxyType(dict(pair for pair in map(_mcp_tool_result_pair, blocks) if pair is not None))


def _assistant_runs(roles: Sequence[object]) -> tuple[int, ...]:
    return tuple(
        accumulate(
            int(role == "assistant" and (index == 0 or roles[index - 1] != "assistant"))
            for index, role in enumerate(roles)
        )
    )


def _first_uses(turns: Sequence[_AssistantTurn]) -> Mapping[str, _AssistantTurn]:
    return MappingProxyType(
        dict(chain.from_iterable(((use_id, turn) for use_id in turn.use_ids) for turn in reversed(turns)))
    )


def _later_result_turn(turns: Sequence[_AssistantTurn], use: _AssistantTurn, use_id: str) -> _AssistantTurn | None:
    return next((turn for turn in turns if turn.message_index > use.message_index and use_id in turn.results), None)


def _grouped_by_message(pairs: Iterable[tuple[int, str]]) -> Mapping[int, frozenset[str]]:
    listed: Final = tuple(pairs)
    return MappingProxyType(
        {
            index: frozenset(use_id for at, use_id in listed if at == index)
            for index in frozenset(at for at, _ in listed)
        }
    )


def _plan(turns: Sequence[_AssistantTurn]) -> McpReplayPlan:
    first_uses: Final = _first_uses(turns)
    unanswered_in_their_message: Final = MappingProxyType(
        {use_id: use for use_id, use in first_uses.items() if use_id not in use.results}
    )
    lone_result_messages: Final = MappingProxyType(
        {
            use_id: later.message_index
            for use_id, use in unanswered_in_their_message.items()
            if (later := _later_result_turn(turns, use, use_id)) is not None
        }
    )
    last_run: Final = turns[-1].run
    repeated_uses: Final = chain.from_iterable(
        ((turn.message_index, use_id) for use_id in turn.use_ids if first_uses[use_id].run < turn.run) for turn in turns
    )
    never_answered_uses: Final = (
        (use.message_index, use_id)
        for use_id, use in unanswered_in_their_message.items()
        if use_id not in lone_result_messages and use.run != last_run
    )
    return McpReplayPlan(
        dropped_ids_by_message=_grouped_by_message(chain(repeated_uses, never_answered_uses)),
        lone_results_by_message=MappingProxyType(
            {
                turn.message_index: tuple(
                    result
                    for use_id, result in turn.results.items()
                    if lone_result_messages.get(use_id) == turn.message_index
                )
                for turn in turns
            }
        ),
        answered_later_ids=frozenset(lone_result_messages),
    )


def _plan_dropping_every_mcp_block(turns: Sequence[_AssistantTurn]) -> McpReplayPlan:
    return McpReplayPlan(
        dropped_ids_by_message=MappingProxyType(
            {turn.message_index: frozenset(chain(turn.use_ids, turn.results)) for turn in turns}
        ),
        lone_results_by_message=MappingProxyType({}),
        answered_later_ids=_NO_IDS,
    )


def mcp_replay_plan(messages: Sequence[object], *, replays_mcp_calls: bool) -> McpReplayPlan:
    roles: Final = tuple(message_field(message, "role") for message in messages)
    runs: Final = _assistant_runs(roles)
    turns: Final = tuple(
        _AssistantTurn(index, runs[index], _mcp_use_ids(message), _mcp_tool_results(message))
        for index, message in enumerate(messages)
        if roles[index] == "assistant"
    )
    if not any(turn.use_ids or turn.results for turn in turns):
        return _NOTHING_TO_CHANGE
    return _plan(turns) if replays_mcp_calls else _plan_dropping_every_mcp_block(turns)


def holds_mcp_blocks(anthropic_messages: Sequence[object]) -> bool:
    return any(
        message_field(block, "type") in _MCP_BLOCK_TYPES
        for block in chain.from_iterable(parts_of(message_field(message, "content")) for message in anthropic_messages)
    )
