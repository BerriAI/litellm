"""Slots H1 and H2: pass-through, vector store and search tool credentials reach only their upstream.

Each test boots an owned proxy whose config declares all three credentials against one
recording upstream double:

- H1: a pass-through endpoint whose ``Authorization`` header is ``Bearer os.environ/<name>``,
  with the canary in that environment variable;
- H2: an OpenAI vector store in ``vector_store_registry`` with the canary as ``api_key``;
- H2S: a Perplexity search tool in ``search_tools`` with the canary as ``api_key``.

The test sends one request through the slot's route, and the upstream answers 200 or, when the
request carries ``UPSTREAM_REJECT``, 401. Positive control: the upstream must receive
``Authorization: Bearer <canary>`` on the request carrying the marker, or the test fails before
sweeping. Sensitivity control: the marker must be reported where stored prompts belong. Then no
sweep may find any of the three canaries.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, Literal

import httpx
import pytest
from integration._support.client import Scenario, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from integration.security._canary import MARKER, Canary, canary
from integration.security._sinks import CONFIG_MODEL, GENERIC_SINK, Caller, Recorder, Rig, canary_rig, settle
from integration.security._sweeps import assert_marker_seen, assert_no_hits, record_route_sweep, sweep_all

Outcome = Literal["success", "upstream_401"]
PASS_THROUGH_ROUTE: Final = "/canary-pass-through"
PASS_THROUGH_ENV: Final = "CANARY_PASS_THROUGH_KEY"
VECTOR_STORE_ID: Final = "canary-vector-store"
SEARCH_TOOL: Final = "canary-search-tool"
UPSTREAM_REJECT: Final = "canary-upstream-reject"
SLOTS: Final = ("H1", "H2", "H2S")
SLACK: Final = timedelta(seconds=5)


def _upstream(request: Request) -> Reply:
    """Pass-through, OpenAI vector store search and Perplexity search double."""
    if UPSTREAM_REJECT.encode() in request.body:
        return Reply(status=401, body=b'{"error":"invalid credentials"}')
    body: Final = json.loads(request.body or b"{}")
    query: Final = str(body.get("query", ""))
    if request.target.startswith("/v1/vector_stores/"):
        return Reply(
            body=json.dumps(
                {
                    "object": "vector_store.search_results.page",
                    "search_query": [query],
                    "data": [
                        {
                            "file_id": "file-canary",
                            "filename": "canary.txt",
                            "score": 0.9,
                            "attributes": {},
                            "content": [{"type": "text", "text": query}],
                        }
                    ],
                    "has_more": False,
                    "next_page": None,
                }
            ).encode()
        )
    if request.target == "/search":
        return Reply(
            body=json.dumps({"results": [{"title": "canary", "url": "https://example.com", "snippet": query}]}).encode()
        )
    return Reply(body=json.dumps({"received": body}).encode())


@dataclass(frozen=True, slots=True)
class Upstreamed:
    rig: Rig
    upstream: Recorder
    canaries: Mapping[str, Canary]


@pytest.fixture
def rigged(tmp_path: Path) -> Iterator[Upstreamed]:
    """One owned proxy per test: the H credentials live in its config and environment."""
    canaries: Final = {slot: canary(slot) for slot in SLOTS}
    with wire_server(_upstream) as wire:

        def configure(config: dict[str, object], provider_url: str) -> None:
            general: Final = config["general_settings"]
            assert isinstance(general, dict)
            general["pass_through_endpoints"] = [
                {
                    "path": PASS_THROUGH_ROUTE,
                    "target": wire.url + "/pass-through",
                    "headers": {"Authorization": f"Bearer os.environ/{PASS_THROUGH_ENV}"},
                    "auth": True,
                }
            ]
            config["vector_store_registry"] = [
                {
                    "vector_store_name": VECTOR_STORE_ID,
                    "litellm_params": {
                        "vector_store_id": VECTOR_STORE_ID,
                        "custom_llm_provider": "openai",
                        "api_key": canaries["H2"].value,
                        "api_base": wire.url + "/v1",
                    },
                }
            ]
            config["search_tools"] = [
                {
                    "search_tool_name": SEARCH_TOOL,
                    "litellm_params": {
                        "search_provider": "perplexity",
                        "api_key": canaries["H2S"].value,
                        "api_base": wire.url,
                    },
                }
            ]

        with canary_rig(tmp_path, configure=configure, environment={PASS_THROUGH_ENV: canaries["H1"].value}) as rig:
            yield Upstreamed(rig, Recorder(wire), canaries)


def _caller(scenario: Scenario) -> Caller:
    team: Final = scenario.team(metadata={"allowed_passthrough_routes": [PASS_THROUGH_ROUTE]})
    user: Final = scenario.user(user_role="internal_user")
    scenario.gateway.post("/team/member_add", {"team_id": team, "member": {"user_id": user, "role": "user"}})
    key: Final = scenario.key(team_id=team, user_id=user, models=[CONFIG_MODEL])
    return Caller(team, user, key)


def _send(rig: Rig, slot: str, key: str, text: str) -> httpx.Response:
    if slot == "H1":
        return rig.proxy.request("POST", PASS_THROUGH_ROUTE, {"text": text}, key=key)
    if slot == "H2":
        return rig.proxy.request("POST", f"/v1/vector_stores/{VECTOR_STORE_ID}/search", {"query": text}, key=key)
    return rig.proxy.request("POST", f"/v1/search/{SEARCH_TOOL}", {"query": text}, key=key)


def _spend_row(marker: Canary, since: datetime) -> str:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE "startTime" >= %s AND proxy_server_request::text LIKE %s',
            (since.astimezone(UTC).replace(tzinfo=None) - SLACK, f"%{marker.core}%"),
        ),
        lambda found: len(found) == 1,
        seconds=70,
    )
    return str(rows[0]["request_id"])


@pytest.mark.timeout(240)  # full S1/S2 walk: every table and ~400 GET routes as two callers
@pytest.mark.parametrize("outcome", ["success", "upstream_401"])
@pytest.mark.parametrize("slot", SLOTS)
def test_upstream_credential_reaches_only_its_upstream(
    rigged: Upstreamed, slot: str, outcome: Outcome, request: pytest.FixtureRequest
) -> None:
    rig: Final = rigged.rig
    credential: Final = rigged.canaries[slot]
    marker: Final = canary(MARKER)
    text: Final = f"slot {slot} {marker.value}" + (f" {UPSTREAM_REJECT}" if outcome == "upstream_401" else "")
    started: Final = datetime.now(UTC)
    with rig.proxy.scenario() as scenario:
        caller: Final = _caller(scenario)
        response: Final = _send(rig, slot, caller.key, text)
        assert response.status_code == (200 if outcome == "success" else 401), response.text
        delivered: Final = rigged.upstream.carrying(marker.core)
        assert [received.headers.get("authorization") for received in delivered] == [f"Bearer {credential.value}"], (
            f"Positive control: the upstream never received the {slot} canary"
        )
        request_id: Final = _spend_row(marker, started)
        delivers_to_sink: Final = not (slot == "H1" and outcome == "upstream_401")
        if delivers_to_sink:
            settle(rig, request_id, marker)

        report: Final = sweep_all(
            rig.proxy,
            (marker, *rigged.canaries.values()),
            responses=(response,),
            sinks={name: sink.requests() for name, sink in rig.sinks.items()},
            ids={
                "request_id": request_id,
                "team_id": caller.team_id,
                "user_id": caller.user_id,
                "vector_store_id": VECTOR_STORE_ID,
                "search_tool_name": SEARCH_TOOL,
                "model": CONFIG_MODEL,
                "model_id": CONFIG_MODEL,
            },
            callers=caller.callers(rig),
            own_headers=rig.own_headers,
            since=started,
        )
        record_route_sweep(report.routes, request.node.nodeid)
        assert_marker_seen(report, {"S2": f"GET /spend/logs?request_id={request_id} as admin -> 200"})
        assert_marker_seen(
            report,
            {
                "S1": "LiteLLM_SpendLogs.proxy_server_request",
                "S2": f"GET /spend/logs/ui/{request_id} as admin -> 200",
                **({"S3": "response[0] POST"} if outcome == "success" else {}),
                **({"S4": f"{GENERIC_SINK}["} if delivers_to_sink else {}),
            },
        )
        assert_no_hits(report.credential_hits(), f"slot {slot}, {outcome}")
