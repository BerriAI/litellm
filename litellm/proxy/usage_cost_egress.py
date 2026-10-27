"""What a client reads in a streamed ``usage.cost``: litellm's bill, written once on the way out of the proxy, onto
a copy, so the final usage frame and the spend row agree. Inside the pipeline ``usage.cost`` stays what the
upstream reported."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Final, TypeVar

from pydantic import BaseModel

from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse
from litellm.types.utils import ModelResponse, ModelResponseStream, Usage

Pricer = Callable[[ModelResponse | ResponsesAPIResponse], float | None]
"""Prices a reply the way the request's logging object bills it (deployment rates, discount, margin)."""

_RESPONSES_USAGE_EVENTS: Final = frozenset({"response.completed", "response.incomplete"})
_ReplyT = TypeVar("_ReplyT", bound=BaseModel)


def _billable(cost: object) -> float | None:
    if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
        return None
    return float(cost)


def _client_usage_cost(upstream_cost: object, bill: object, *, wanted: bool) -> float | None:
    """The bill, when the upstream put a number in usage.cost or the event carries one regardless (``wanted``).
    None leaves the usage as the upstream sent it."""
    reported: Final = isinstance(upstream_cost, (int, float)) and not isinstance(upstream_cost, bool)
    return _billable(bill) if reported or wanted else None


def _with_usage_cost(reply: _ReplyT, usage: Usage | ResponseAPIUsage, cost: float) -> _ReplyT:
    return reply.model_copy(update={"usage": usage.model_copy(update={"cost": cost})})


def with_client_stream_usage_cost(chunk: BaseModel, price: Pricer) -> BaseModel:
    """A streamed event as the client gets it. Only the usage-bearing events are priced: the chat chunk carrying
    ``usage``, rewritten where the upstream reported a cost, and the Responses ``response.completed`` /
    ``response.incomplete`` events, which carry the bill whether or not the upstream reported one."""
    if isinstance(chunk, ModelResponseStream):
        usage: Final = getattr(chunk, "usage", None)
        if not isinstance(usage, Usage):
            return chunk
        priced: Final = ModelResponse(usage=usage, service_tier=getattr(chunk, "service_tier", None))
        cost: Final = _client_usage_cost(getattr(usage, "cost", None), price(priced), wanted=False)
        return chunk if cost is None else _with_usage_cost(chunk, usage, cost)
    if getattr(chunk, "type", None) not in _RESPONSES_USAGE_EVENTS:
        return chunk
    response: Final = getattr(chunk, "response", None)
    if not isinstance(response, ResponsesAPIResponse) or not isinstance(response.usage, ResponseAPIUsage):
        return chunk
    response_cost: Final = _client_usage_cost(getattr(response.usage, "cost", None), price(response), wanted=True)
    if response_cost is None:
        return chunk
    return chunk.model_copy(update={"response": _with_usage_cost(response, response.usage, response_cost)})
