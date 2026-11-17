"""Shared pieces for the ``test_dec1_capacity3_*`` contracts (model capacity, round 3).

Round 3 targets what the Rust port of the dynamic rate limiter v3 (draft PR #51540, stacked on the limiter port
#51537) has to carry over from the four Python changes ported into it: #51705 (customer organizations and keys
borrow idle capacity, class shares split by key, customer organization and customer member, websocket turns fail
closed, the empty hold on a fallback target a borrower has no limit on), #51952 (a class share counts requests and
tokens while the model is idle), #51996 (a refactor of the settlement flattening, behaviour neutral) and #52081
(priority counters keep counting requests after their tokens run out). Every literal is read from
``litellm/proxy/hooks/dynamic_rate_limiter_v3.py`` on the branch.

Everything else comes from the earlier capacity rounds by import: the capacity proxy and model entries from
``dec1_capacity_support.py``, Redis counter reads from ``authorization/dec1_limcap4_support.py`` and the customer
hierarchy helpers from ``authorization/dec1_limcap2_support.py``. Counter reads need ``capacity_proxy(shared=True)``
so the proxy writes to the Redis ``shared_redis()`` reads.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import websockets
from pydantic import JsonValue

from tests.integration._support.client import Gateway, Scenario
from tests.integration._support.responses_ws_peer import PROVIDER_MODEL, create, turn
from tests.integration.authorization.dec1_limcap2_support import CUSTOMERS
from tests.integration.authorization.dec1_limits_support import counter_key
from tests.integration.routing.dec1_capacity_support import PROVIDER_KEY

CLASS_SHARE_SCOPE: Final = "priority_class_share_saturated"
HOLDER_SEPARATOR: Final = "\x1f"
WS_TURN_TOKENS: Final = 7


def class_share_value(model: str, holder: str) -> str:
    """``counter_value`` (priority_class_share.py:57): ``<len(model)>:<model>:<holder>``. A team holder is the raw
    team id, a key is ``key\\x1f<hash>``, a customer organization ``customer_org\\x1f<id>`` and a customer member
    ``customer_member\\x1f<id>`` (priority_class_share.py:43-50, dynamic_rate_limiter_v3.py:679)."""
    return f"{len(model)}:{model}:{holder}"


def class_share_counter(model: str, holder: str, kind: str) -> str:
    return counter_key(CLASS_SHARE_SCOPE, class_share_value(model, holder), kind)


def holder(kind: str, entity_id: str) -> str:
    """A non-team class share holder, for example ``holder("customer_org", org_id)``."""
    return f"{kind}{HOLDER_SEPARATOR}{entity_id}"


def borrower_scope(kind: str, *, capped: bool = False) -> str:
    """``model_per_<kind>_saturated`` or ``model_per_<kind>_max_borrow_multiple`` (dynamic_rate_limiter_v3.py:200-205),
    where ``kind`` is ``team``, ``key`` or ``customer_org``."""
    return f"model_per_{kind}_max_borrow_multiple" if capped else f"model_per_{kind}_saturated"


def borrower_value(entity_id: str, model: str) -> str:
    """A borrower's share counter value, ``<entity_id>:<group>`` (dynamic_rate_limiter_v3.py:785)."""
    return f"{entity_id}:{model}"


def ws_entry(group: str, peer_url: str, **params: JsonValue) -> dict[str, JsonValue]:
    """One OpenAI deployment of ``group`` whose Responses WebSocket turns are relayed to the scripted TLS peer at
    ``peer_url`` (which already ends in ``/v1``)."""
    return {
        "model_name": group,
        "litellm_params": {
            "model": f"openai/{PROVIDER_MODEL}",
            "api_key": PROVIDER_KEY,
            "api_base": peer_url,
            **params,
        },
        "model_info": {"id": f"{group}-{uuid.uuid4().hex[:6]}"},
    }


@dataclass(frozen=True, slots=True)
class SocketOutcome:
    """Every turn's events in order and, when the proxy closed the socket, its close code and reason."""

    turns: tuple[tuple[dict[str, JsonValue], ...], ...]
    close_code: int | None
    close_reason: str | None


def _socket_url(gateway: Gateway, model: str) -> str:
    return f"{str(gateway.client.base_url).rstrip('/').replace('http://', 'ws://')}/v1/responses?model={model}"


async def _turns(
    gateway: Gateway, key: str, model: str, count: int, headers: Mapping[str, str] | None
) -> SocketOutcome:
    async with websockets.connect(
        _socket_url(gateway, model),
        additional_headers={"Authorization": f"Bearer {key}", **(headers or {})},
        open_timeout=10,
    ) as connection:
        received: tuple[tuple[dict[str, JsonValue], ...], ...] = ()  # rebind-ok: grows one turn at a time
        try:
            for _ in range(count):
                received = (*received, await turn(connection, create(model, f"capacity3 {uuid.uuid4().hex}")))
        except websockets.ConnectionClosed as closed:
            return SocketOutcome(
                received, closed.rcvd.code if closed.rcvd else None, closed.rcvd.reason if closed.rcvd else None
            )
        return SocketOutcome(received, None, None)


def socket_turns(
    gateway: Gateway, key: str, model: str, count: int, headers: Mapping[str, str] | None = None
) -> SocketOutcome:
    """Open one Responses WebSocket on ``/v1/responses?model=<model>``, send ``count`` ``response.create`` frames one
    after another and collect each turn's events until a terminal event."""
    return asyncio.run(_turns(gateway, key, model, count, headers))


async def _refused_at_open(gateway: Gateway, key: str, model: str) -> tuple[dict[str, JsonValue], int, str]:
    async with websockets.connect(
        _socket_url(gateway, model), additional_headers={"Authorization": f"Bearer {key}"}, open_timeout=10
    ) as connection:
        await connection.send(create(model, "capacity3 refused at open"))
        frame: Final = json.loads(await asyncio.wait_for(connection.recv(), timeout=20))
        try:
            await asyncio.wait_for(connection.recv(), timeout=20)
        except websockets.ConnectionClosed as closed:
            assert closed.rcvd is not None
            return frame, closed.rcvd.code, closed.rcvd.reason
        raise AssertionError("The proxy kept the refused socket open")


def refused_at_open(gateway: Gateway, key: str, model: str) -> tuple[dict[str, JsonValue], int, str]:
    """The one frame the proxy sends before closing a socket it refused while opening it, plus the close code and
    reason (response_api_endpoints/endpoints.py:2700-2706)."""
    return asyncio.run(_refused_at_open(gateway, key, model))


def turn_types(outcome: SocketOutcome) -> list[str]:
    """The terminal event type of each turn: ``response.completed`` or ``error``."""
    return [str(events[-1].get("type")) for events in outcome.turns]


def add_member(gateway: Gateway, team_id: str) -> str:
    """One more customer member under an existing customer team, created through ``/management/v1``."""
    member: Final = f"capacity3-member-{uuid.uuid4().hex[:10]}"
    created: Final = gateway.request(
        "POST", f"{CUSTOMERS}/customer_members", {"customer_member_id": member, "parent_team_ids": [team_id]}
    )
    assert created.status_code == 201, created.text
    return member


def interactive_key(scenario: Scenario, **fields: JsonValue) -> str:
    return scenario.key(metadata={"priority": "interactive"}, **fields)
