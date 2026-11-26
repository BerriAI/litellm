from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, TypeAlias

from litellm._uuid import uuid
from litellm.litellm_core_utils.allowed_tools_tool_choice import GatewayTools
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.responses.mcp.mcp_streaming_iterator import output_items
from litellm.responses.mcp.openai_mcp_items import OpenAIMCPItems, openai_output, ran_round_output
from litellm.responses.mcp.round_billing import (
    billed_for_every_round,
    first_round_billed_on_failure,
    summed_cost_breakdown,
)
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.types.llms.openai import ResponseInputParam, ResponsesAPIResponse
from litellm.types.utils import CostBreakdown

if TYPE_CHECKING:
    from litellm.responses.mcp.litellm_proxy_mcp_handler import MCPToolResult

ToolExecutor: TypeAlias = Callable[[Sequence[object]], Awaitable[Sequence["MCPToolResult"]]]
FollowUp: TypeAlias = Callable[
    [Sequence[object], bool], Awaitable[ResponsesAPIResponse | BaseResponsesAPIStreamingIterator]
]
BreakdownRecorder: TypeAlias = Callable[[CostBreakdown | None], None]


@dataclass(frozen=True, slots=True)
class ClaimedRounds:
    """One request's tool loop for the hosted MCP tools the gateway runs. `follow_up` sends the next round, with the
    tools when its flag is True; the last allowed round goes without them so the model answers in text."""

    items: OpenAIMCPItems
    execute: ToolExecutor
    follow_up: FollowUp
    record_breakdown: BreakdownRecorder
    logging_obj: object
    max_rounds: int
    tool_choice: object
    gateway: GatewayTools


def _call_details(tool_call: object) -> tuple[str, str, str]:
    from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler

    name, arguments, call_id = LiteLLM_Proxy_MCP_Handler.extract_tool_call_details(tool_call)
    return name or "", arguments or "{}", call_id or ""


def _mcp_call(result: "MCPToolResult", arguments_by_call: Mapping[str, str]) -> Mapping[str, object]:
    call_id: Final = result["tool_call_id"] or ""
    return {
        "id": f"mcp_{uuid.uuid4().hex}",
        "type": "mcp_call",
        "name": result["name"] or "",
        "arguments": arguments_by_call.get(call_id, "{}"),
        "output": result["result"],
        "error": result.get("error"),
    }


def _round_items(
    response: ResponsesAPIResponse, tool_calls: Sequence[object], results: Sequence["MCPToolResult"]
) -> tuple[object, ...]:
    arguments_by_call: Final = {call_id: arguments for _, arguments, call_id in map(_call_details, tool_calls)}
    ran_calls: Final = MappingProxyType(
        {result["tool_call_id"]: _mcp_call(result, arguments_by_call) for result in results if result["tool_call_id"]}
    )
    return ran_round_output(output_items(response), ran_calls)


def _answer(
    rounds: ClaimedRounds, response: ResponsesAPIResponse, composed: tuple[object, ...], round_output: Sequence[object]
) -> ResponsesAPIResponse:
    return response.model_copy(update={"output": list(openai_output(rounds.items, (*composed, *round_output)))})


def _handed_back(
    rounds: ClaimedRounds, response: ResponsesAPIResponse, composed: tuple[object, ...]
) -> ResponsesAPIResponse:
    return response.model_copy(update={"output": [*openai_output(rounds.items, composed), *output_items(response)]})


async def _held_round_output(
    rounds: ClaimedRounds, response: ResponsesAPIResponse, claimed_calls: Sequence[object]
) -> tuple[object, ...]:
    with first_round_billed_on_failure(rounds.logging_obj, response):
        results: Final = await rounds.execute(claimed_calls)
    return _round_items(response, claimed_calls, results)


def _breakdown(logging_obj: object) -> CostBreakdown | None:
    return logging_obj.cost_breakdown if isinstance(logging_obj, LiteLLMLoggingObj) else None


async def _next_round(
    rounds: ClaimedRounds,
    response: ResponsesAPIResponse,
    breakdown: CostBreakdown | None,
    history: tuple[object, ...],
    composed: tuple[object, ...],
    round_number: int,
) -> ResponsesAPIResponse | BaseResponsesAPIStreamingIterator:
    from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler

    requested_calls: Final = LiteLLM_Proxy_MCP_Handler.extract_tool_calls_from_response(response)
    tool_calls: Final = LiteLLM_Proxy_MCP_Handler.tool_calls_to_run(requested_calls, rounds.tool_choice, rounds.gateway)
    if requested_calls and not tool_calls:
        return _handed_back(rounds, response, composed)
    claimed_calls: Final = tuple(call for call in tool_calls if _call_details(call)[0] in rounds.items.names)
    if not claimed_calls:
        return _answer(rounds, response, composed, output_items(response))
    if round_number > rounds.max_rounds:
        return _handed_back(rounds, response, composed)
    if len(claimed_calls) < len(tool_calls):
        return _answer(rounds, response, composed, await _held_round_output(rounds, response, claimed_calls))
    with first_round_billed_on_failure(rounds.logging_obj, response):
        results: Final = await rounds.execute(tool_calls)
        follow_up_input: Final = (
            *history,
            *LiteLLM_Proxy_MCP_Handler.create_follow_up_input(
                response=response, tool_results=results, preserve_reasoning=True
            ),
        )
        follow_up: Final = await rounds.follow_up(follow_up_input, round_number < rounds.max_rounds)
    if not isinstance(follow_up, ResponsesAPIResponse):
        return follow_up
    follow_up_breakdown: Final = _breakdown(rounds.logging_obj)
    summed: Final = summed_cost_breakdown(
        first=response, first_breakdown=breakdown, final=follow_up, final_breakdown=follow_up_breakdown
    )
    rounds.record_breakdown(summed)
    return await _next_round(
        rounds,
        billed_for_every_round(
            first=response, first_breakdown=breakdown, final=follow_up, final_breakdown=follow_up_breakdown
        ),
        summed,
        follow_up_input,
        (*composed, *_round_items(response, tool_calls, results)),
        round_number + 1,
    )


def _initial_history(original_input: str | ResponseInputParam | None) -> tuple[object, ...]:
    if not original_input:
        return ()
    if isinstance(original_input, str):
        return ({"type": "message", "role": "user", "content": original_input},)
    return tuple(original_input)


async def run_claimed_rounds(
    rounds: ClaimedRounds,
    first: ResponsesAPIResponse,
    first_breakdown: CostBreakdown | None,
    original_input: str | ResponseInputParam | None,
) -> ResponsesAPIResponse | BaseResponsesAPIStreamingIterator:
    """Run every tool call the model makes on the gateway's MCP tools, round after round up to `max_rounds`, and
    answer with OpenAI's items: mcp_list_tools first, each executed call as an mcp_call after its round's other items,
    and one bill for all rounds. A round that also calls a tool the gateway does not run ends the loop: the gateway
    runs its own calls and hands the others back to the caller as function_call items ahead of those mcp_call items.
    Calls the model still makes after `max_rounds` are handed back unrun the same way."""
    return await _next_round(rounds, first, first_breakdown, _initial_history(original_input), (), 1)
