import asyncio
from collections.abc import Awaitable, Iterable, Mapping
from itertools import chain
from types import MappingProxyType
from typing import Final

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.prompt_templates.mid_conversation_system import message_field, parts_of
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import (
    ChatCompletionMessageCustomToolCall,
    ChatCompletionMessageToolCall,
    Choices,
    ModelResponse,
)


def _strings(values: Iterable[object]) -> frozenset[str]:
    return frozenset(value for value in values if isinstance(value, str))


def _claimed_calls(tool_calls: object) -> tuple[object, ...]:
    wrapped: Final = message_field(tool_calls, "tool_calls")
    return parts_of(tool_calls if wrapped is None else wrapped)


async def _calls_a_gate_claims(gate: Awaitable[tuple[bool, object]]) -> tuple[object, ...]:
    try:
        should_run, tool_calls = await gate
    except Exception as e:
        verbose_logger.debug("LiteLLM.AgenticLoopRefused: skipping a gate that failed with %s", type(e).__name__)
        return ()
    return _claimed_calls(tool_calls) if should_run else ()


async def every_claimed_tool_call(
    refused_tool_calls: object, other_gates: Iterable[Awaitable[tuple[bool, object]]]
) -> Mapping[str, tuple[object, ...]]:
    claims: Final = await asyncio.gather(*(_calls_a_gate_claims(gate) for gate in other_gates))
    return MappingProxyType({"tool_calls": (*_claimed_calls(refused_tool_calls), *chain.from_iterable(claims))})


def _call_ids(call: object) -> frozenset[str]:
    return _strings((message_field(call, "id"), message_field(call, "call_id")))


def refused_tool_call_identifiers(tool_calls: object) -> tuple[frozenset[str], frozenset[str]]:
    calls: Final = _claimed_calls(tool_calls)
    ids: Final = frozenset(chain.from_iterable(_call_ids(call) for call in calls))
    names: Final = _strings(message_field(call, "name") for call in calls if not _call_ids(call))
    return ids, names


def is_refused_call(call_id: object, name: object, refused_ids: frozenset[str], refused_names: frozenset[str]) -> bool:
    return (isinstance(call_id, str) and call_id in refused_ids) or name in refused_names


def is_refused_item(
    item: object, item_type: str, id_field: str, refused_ids: frozenset[str], refused_names: frozenset[str]
) -> bool:
    if message_field(item, "type") != item_type:
        return False
    return is_refused_call(message_field(item, id_field), message_field(item, "name"), refused_ids, refused_names)


def finalize_refused_responses_turn(response: ResponsesAPIResponse, tool_calls: object) -> ResponsesAPIResponse:
    refused_ids, refused_names = refused_tool_call_identifiers(tool_calls)
    kept: Final = [
        item
        for item in parts_of(message_field(response, "output"))
        if not is_refused_item(item, "function_call", "call_id", refused_ids, refused_names)
    ]
    return response.model_copy(update={"output": kept})


def _is_refused_chat_tool_call(call: object, refused_ids: frozenset[str], refused_names: frozenset[str]) -> bool:
    name: Final = message_field(message_field(call, "function"), "name")
    return is_refused_call(message_field(call, "id"), name, refused_ids, refused_names)


def _at_position(
    call: ChatCompletionMessageToolCall | ChatCompletionMessageCustomToolCall, position: int
) -> ChatCompletionMessageToolCall | ChatCompletionMessageCustomToolCall:
    return call if message_field(call, "index") is None else call.model_copy(update={"index": position})


def _finalized_choice(choice: object, refused_ids: frozenset[str], refused_names: frozenset[str]) -> object:
    if not isinstance(choice, Choices) or not choice.message.tool_calls:
        return choice
    surviving: Final = (
        call for call in choice.message.tool_calls if not _is_refused_chat_tool_call(call, refused_ids, refused_names)
    )
    kept: Final = [_at_position(call, position) for position, call in enumerate(surviving)]
    finish_reason: Final = "stop" if not kept and choice.finish_reason == "tool_calls" else choice.finish_reason
    return choice.model_copy(
        update={
            "message": choice.message.model_copy(update={"tool_calls": kept or None}),
            "finish_reason": finish_reason,
        }
    )


def finalize_refused_chat_turn(response: ModelResponse, tool_calls: object) -> ModelResponse:
    refused_ids, refused_names = refused_tool_call_identifiers(tool_calls)
    return response.model_copy(
        update={"choices": [_finalized_choice(choice, refused_ids, refused_names) for choice in response.choices]}
    )
