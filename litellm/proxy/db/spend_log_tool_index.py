"""
Tool usage tracking for the dashboard.

At request time the spend writer builds one ToolUsageTransaction per request that
invoked tools (MCP namespaced tool name plus response tool_calls; declared-but-not-
invoked tools are excluded) and queues it on the prisma client. The spend-log flush
job drains the queue into LiteLLM_SpendLogToolIndex (per-request drill-down) and
LiteLLM_DailyToolSpend (the per-day rollup the Cost Optimization card reads) in a
single transaction, so a failed flush never leaves a partial rollup increment.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import groupby
from typing import TYPE_CHECKING, Any, Final

from litellm.proxy._types import DB_RETRY_SAFE_ERROR_TYPES

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


@dataclass(frozen=True, slots=True)
class ToolUsageTransaction:
    request_id: str
    date: str
    start_time: datetime
    tool_names: tuple[str, ...]
    spend: float
    total_tokens: int


def response_tool_call_names(completion_response: Any) -> tuple[str, ...]:
    """Tool names invoked in a completion response, in call order, for any response
    surface get_tool_calls_from_response understands (chat completions, Responses
    API output items, Anthropic Messages tool_use blocks). Reads every choice of
    an ``n>1`` chat response: each choice cost money and its tool calls ran."""
    if completion_response is None or isinstance(completion_response, Exception):
        return ()
    from litellm.litellm_core_utils.prompt_templates.factory import (
        get_tool_calls_from_response,
    )

    return tuple(
        stripped
        for tool_call in get_tool_calls_from_response(completion_response, include_all_choices=True)
        if isinstance(name := tool_call.get("name"), str) and (stripped := name.strip())
    )


def build_tool_usage_transaction(
    request_id: str,
    start_time_iso: str,
    mcp_namespaced_tool_name: str | None,
    spend: float,
    total_tokens: int,
    completion_response: Any,
    realtime_tool_calls: Any = None,
) -> ToolUsageTransaction | None:
    """None when the request invoked no tools. Realtime sessions carry invoked
    tools in kwargs["realtime_tool_calls"] (OpenAI tool_calls shape) rather than
    on a response object, so they are normalized through the same owner by
    wrapping them in the chat-completion shape. Date derivation must match the
    daily spend writer's ``startTime.split("T")[0]`` so rollup rows land in the
    same UTC day bucket as LiteLLM_DailyUserSpend."""
    mcp_names: Final = (
        (mcp_namespaced_tool_name.strip(),) if mcp_namespaced_tool_name and mcp_namespaced_tool_name.strip() else ()
    )
    realtime_names: Final = (
        response_tool_call_names({"choices": [{"message": {"tool_calls": realtime_tool_calls}}]})
        if realtime_tool_calls
        else ()
    )
    tool_names: Final = tuple(dict.fromkeys(mcp_names + response_tool_call_names(completion_response) + realtime_names))
    if not tool_names:
        return None
    try:
        start_time: Final = datetime.fromisoformat(start_time_iso.replace("Z", "+00:00"))
    except ValueError:
        return None
    return ToolUsageTransaction(
        request_id=request_id,
        date=start_time_iso.split("T")[0],
        start_time=start_time if start_time.tzinfo else start_time.replace(tzinfo=timezone.utc),
        tool_names=tool_names,
        spend=spend,
        total_tokens=total_tokens,
    )


async def flush_tool_usage_transactions(
    prisma_client: PrismaClient,
    transactions: Sequence[ToolUsageTransaction],
    n_retry_times: int = 3,
) -> None:
    """Write index rows and rollup upserts for a drained queue batch in one
    transaction. Retries only ConnectError, the one failure that proves the
    statements never reached the database. Post-send failures (Read timeouts
    and errors) are ambiguous and are NOT retried: the engine can abandon the
    transaction open on the pooled connection, so a retry's statements stack
    into the same transaction and one commit applies both increment sets.
    Ambiguous failures drop the batch; the caller logs it at error. Callers
    must not add their own retry around this function."""
    if not transactions:
        return

    index_rows: Final = [
        {"request_id": txn.request_id, "tool_name": tool_name, "start_time": txn.start_time}
        for txn in transactions
        for tool_name in txn.tool_names
    ]
    per_tool_day: Final = sorted(
        ((txn.date, tool_name, txn.spend, txn.total_tokens) for txn in transactions for tool_name in txn.tool_names),
        key=lambda entry: (entry[0], entry[1]),
    )

    for attempt in range(n_retry_times + 1):
        try:
            async with prisma_client.db.batch_() as batcher:
                batcher.litellm_spendlogtoolindex.create_many(data=index_rows, skip_duplicates=True)
                for (date_key, tool_name), grouped in groupby(per_tool_day, key=lambda entry: (entry[0], entry[1])):
                    entries = tuple(grouped)
                    spend = sum(entry[2] for entry in entries)
                    total_tokens = sum(entry[3] for entry in entries)
                    batcher.litellm_dailytoolspend.upsert(
                        where={"date_tool_name": {"date": date_key, "tool_name": tool_name}},
                        data={
                            "create": {
                                "date": date_key,
                                "tool_name": tool_name,
                                "spend": spend,
                                "total_tokens": total_tokens,
                                "request_count": len(entries),
                            },
                            "update": {
                                "spend": {"increment": spend},
                                "total_tokens": {"increment": total_tokens},
                                "request_count": {"increment": len(entries)},
                            },
                        },
                    )
            return
        except DB_RETRY_SAFE_ERROR_TYPES:
            if attempt >= n_retry_times:
                raise
            await asyncio.sleep(2**attempt + random.uniform(0, 1))
