import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Final

import httpx
import pytest
from integration._support.actors import ManagementDatabase
from integration._support.client import Gateway
from integration._support.database import write_rows
from pydantic import JsonValue, TypeAdapter

from litellm.proxy.agent_endpoints.external_cloud_agents.billing import SpendSync, utc_now
from litellm.proxy.agent_endpoints.external_cloud_agents.database import ProxyDatabase
from litellm.proxy.agent_endpoints.external_cloud_agents.models import AgentPage, Connection, SpendReport, SyncResult

ROOT: Final = "/v1/external_cloud_agents"
pytestmark: Final = pytest.mark.shares_worker_state


class BillingVendor:
    def __init__(self, now: datetime) -> None:
        self.now = now
        self.cost = "1.1234567891"
        self.empty = False

    def respond(self, request: httpx.Request) -> httpx.Response:
        assert request.url.host == "billing.example.test"
        assert request.url.path == "/v2/teams/billing-items"
        return httpx.Response(
            200,
            json={
                "data": []
                if self.empty
                else [
                    {
                        "eventTime": (self.now - timedelta(minutes=1)).isoformat(),
                        "chargeCategory": "usage",
                        "principal": {"type": "user", "id": "vendor-billing-member"},
                        "pricing": {"contracted": self.cost},
                        "usage": {"requests": "4"},
                    }
                ],
                "pagination": {"hasMore": False},
            },
        )

    def replace_cost(self) -> None:
        self.cost = "3.25"

    def clear(self) -> None:
        self.empty = True


@pytest.mark.asyncio
async def test_vendor_sync_replaces_real_database_rows_idempotently_and_clears_disappeared_items(
    connected: "Connected",
    db: ManagementDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", connected.gateway.key)
    now: Final = utc_now()
    vendor: Final = BillingVendor(now)
    source_id: Final = connected.connection.source_id
    async with httpx.AsyncClient(transport=httpx.MockTransport(vendor.respond)) as client:
        sync: Final = SpendSync(
            ProxyDatabase(db.prisma.db), client, lambda: now, base_url="https://billing.example.test"
        )
        initial: Final = await sync.save_key(source_id, "billing-test-key")
        assert initial == SyncResult(rows=1)
        again: Final = await sync.run(source_id)
        assert again == initial
        persisted: Final = connected.gateway.rows(
            'SELECT cost_micros AS cost,requests FROM "LiteLLM_ExternalCloudAgentSpend" WHERE source_id=%s',
            (source_id,),
        )
        assert persisted == [{"cost": 1123457, "requests": 4}]
        vendor.replace_cost()
        updated: Final = await sync.run(source_id)
        assert updated == initial
        changed: Final = connected.gateway.rows(
            'SELECT cost_micros AS cost FROM "LiteLLM_ExternalCloudAgentSpend" WHERE source_id=%s', (source_id,)
        )
        assert changed == [{"cost": 3250000}]
        vendor.clear()
        cleared: Final = await sync.run(source_id)
        assert cleared == SyncResult(rows=0)
        assert (
            connected.gateway.rows('SELECT * FROM "LiteLLM_ExternalCloudAgentSpend" WHERE source_id=%s', (source_id,))
            == []
        )
    removed: Final = connected.gateway.request("DELETE", ROOT + f"/sources/{source_id}/admin_api_key")
    assert removed.status_code == 200, removed.text
    missing_key: Final = connected.gateway.request("POST", ROOT + f"/sources/{source_id}/sync_spend")
    assert missing_key.status_code == 400, missing_key.text


@dataclass(frozen=True, slots=True)
class Connected:
    gateway: Gateway
    connection: Connection

    def hook(self, body: dict[str, JsonValue], key: str | None = None) -> httpx.Response:
        if key == "":
            return self.gateway.client.post(self.connection.hook_path, json=body)
        return self.gateway.request(
            "POST", self.connection.hook_path, body, key=self.connection.ingest_key if key is None else key
        )

    def agents(self, **params: str) -> AgentPage:
        response: Final = self.gateway.request(
            "GET", ROOT + "/agents", params={"source_id": self.connection.source_id, **params}
        )
        assert response.status_code == 200, response.text
        return AgentPage.model_validate(response.json())


@pytest.fixture
def connected(gateway: Gateway) -> Iterator[Connected]:
    response: Final = gateway.request("POST", ROOT + "/sources")
    assert response.status_code == 200, response.text
    connection: Final = Connection.model_validate(response.json())
    try:
        yield Connected(gateway, connection)
    finally:
        deleted: Final = gateway.request("DELETE", ROOT + "/sources/" + connection.source_id)
        assert deleted.status_code == 200, deleted.text


def frame(bot: str = "bot", tenant: str = "tenant") -> dict[str, JsonValue]:
    return {
        "session_id": bot,
        "tenant_id": tenant,
        "actor": {"id": "member"},
        "type": "prompt",
        "model": "external-model",
    }


@pytest.mark.parametrize("scope", ["llm_api_routes", "agent_routes"])
def test_admin_owned_inference_keys_cannot_read_connection(connected: Connected, scope: str) -> None:
    route: Final = ROOT + f"/sources/{connected.connection.source_id}/connection"
    with connected.gateway.scenario() as scenario:
        user: Final = scenario.user(user_role="proxy_admin")
        key: Final = scenario.key(user_id=user, allowed_routes=[scope])
        denied: Final = connected.gateway.request("GET", route, key=key)
        assert denied.status_code == 403, denied.text
        assert connected.connection.ingest_key not in denied.text
        allowed: Final = connected.gateway.request("GET", route)
        assert allowed.status_code == 200, allowed.text
        assert Connection.model_validate(allowed.json()) == connected.connection


def test_connect_test_auth_and_content_free_discovery(connected: Connected) -> None:
    probe: Final = connected.hook({**frame(), "source": {"application": "config-test"}})
    assert probe.status_code == 200 and probe.json() == {"action": "allow"}, probe.text
    assert connected.agents().total == 0
    for key in ("", "wrong-key"):
        rejected = connected.hook(frame(), key)
        assert rejected.status_code == 401, rejected.text
    unknown: Final = connected.gateway.request(
        "POST", "/grok_bot/inference_hook/unknown", frame(), key=connected.connection.ingest_key
    )
    assert unknown.status_code == 401, unknown.text
    first: Final = connected.hook({**frame(), "unknown": "future field", "messages": [{"content": "private-prompt"}]})
    assert first.status_code == 200 and first.json() == {"action": "allow"}, first.text
    second: Final = connected.hook(frame())
    assert second.status_code == 200, second.text
    mismatch: Final = connected.hook(frame(tenant="different-tenant"))
    assert mismatch.status_code == 401, mismatch.text
    inventory: Final = connected.agents()
    assert inventory.total == 1
    assert inventory.data[0].request_count == 2
    assert inventory.data[0].last_actor_id == "member"
    assert inventory.data[0].last_model == "external-model"
    stored: Final = connected.gateway.rows(
        'SELECT row_to_json(a)::text AS value FROM "LiteLLM_ExternalCloudAgent" a WHERE source_id=%s',
        (connected.connection.source_id,),
    )
    assert "private-prompt" not in str(stored)
    listed: Final = connected.gateway.request("GET", ROOT + "/sources")
    sources: Final = TypeAdapter(tuple[dict[str, JsonValue], ...]).validate_python(listed.json())
    source: Final = next(row for row in sources if row["source_id"] == connected.connection.source_id)
    assert source["bot_count"] == 1 and source["connected"] is True
    assert source["last_event_at"] is not None
    assert "ingest_key" not in source and "admin_api_key" not in source
    assert connected.connection.ingest_key not in listed.text
    stored_key: Final = connected.gateway.rows(
        'SELECT ingest_key FROM "LiteLLM_ExternalCloudAgentSource" WHERE id=%s', (connected.connection.source_id,)
    )
    assert stored_key[0]["ingest_key"] != connected.connection.ingest_key
    shown: Final = connected.gateway.request("GET", ROOT + f"/sources/{connected.connection.source_id}/connection")
    assert shown.status_code == 200, shown.text
    assert Connection.model_validate(shown.json()) == connected.connection


def test_concurrent_counters_tool_merge_capping_search_and_pagination(connected: Connected) -> None:
    def send(index: int) -> httpx.Response:
        return connected.hook(
            {
                **frame(),
                "type": "tool_call",
                "messages": [
                    {
                        "content": [
                            {
                                "type": "tool_use",
                                "name": f"tool-{index:03}",
                                "tool_info": {"toolset_name": "mcp-tools"},
                            }
                        ]
                    }
                ],
            }
        )

    with ThreadPoolExecutor(max_workers=20) as executor:
        responses: Final = tuple(executor.map(send, range(70)))
    assert tuple(response.status_code for response in responses) == (200,) * 70
    inventory: Final = connected.agents()
    assert inventory.total == 1 and inventory.data[0].request_count == 70
    assert inventory.data[0].tools == ("mcp-tools", *(f"tool-{i:03}" for i in range(49)))
    added: Final = connected.hook(frame(bot="other-bot"))
    assert added.status_code == 200, added.text
    assert connected.agents(page="1", page_size="1").total == 2
    assert len(connected.agents(page="2", page_size="1").data) == 1
    assert connected.agents(search="OTHER").data[0].external_id == "other-bot"
    assert connected.agents(search="missing").total == 0


def test_concurrent_first_frames_bind_one_tenant(connected: Connected) -> None:
    def send(tenant: str) -> tuple[str, httpx.Response]:
        return tenant, connected.hook(frame(bot=tenant, tenant=tenant))

    with ThreadPoolExecutor(max_workers=2) as executor:
        results: Final = tuple(executor.map(send, ("tenant-a", "tenant-b")))
    assert sorted(response.status_code for _, response in results) == [200, 401]
    winner: Final = next(tenant for tenant, response in results if response.status_code == 200)
    assert connected.agents().data[0].external_id == winner


def test_hook_retries_count_each_frame_once_across_concurrent_deliveries(connected: Connected) -> None:
    body: Final = {**frame(), "request_id": "request-one"}
    with ThreadPoolExecutor(max_workers=10) as executor:
        responses: Final = tuple(executor.map(lambda _: connected.hook(body), range(20)))
    assert tuple(response.status_code for response in responses) == (200,) * 20
    assert connected.agents().data[0].request_count == 1
    with ThreadPoolExecutor(max_workers=5) as executor:
        distinct: Final = tuple(
            executor.map(lambda index: connected.hook({**body, "request_id": f"distinct-{index}"}), range(5))
        )
    assert tuple(response.status_code for response in distinct) == (200,) * 5
    assert connected.agents().data[0].request_count == 6
    other: Final = connected.hook({**body, "session_id": "other-bot"})
    assert other.status_code == 200, other.text
    assert connected.agents(search="other-bot").data[0].request_count == 1
    created: Final = connected.gateway.request("POST", ROOT + "/sources")
    assert created.status_code == 200, created.text
    connection: Final = Connection.model_validate(created.json())
    second: Final = Connected(connected.gateway, connection)
    try:
        with ThreadPoolExecutor(max_workers=5) as executor:
            retried: Final = tuple(executor.map(lambda _: second.hook(body), range(10)))
        assert tuple(response.status_code for response in retried) == (200,) * 10
        assert second.agents().data[0].request_count == 1
        assert next(row for row in connected.agents().data if row.external_id == "bot").request_count == 6
    finally:
        deleted: Final = connected.gateway.request("DELETE", ROOT + "/sources/" + connection.source_id)
        assert deleted.status_code == 200, deleted.text


def test_hook_retry_window_evicts_old_ids_and_accepts_frames_without_ids(connected: Connected) -> None:
    for index in range(257):
        response: Final = connected.hook({**frame(), "request_id": f"request-{index}"})
        assert response.status_code == 200, response.text
    replay: Final = connected.hook({**frame(), "request_id": "request-0"})
    assert replay.status_code == 200, replay.text
    assert connected.agents().data[0].request_count == 258
    legacy: Final = connected.hook(frame())
    assert legacy.status_code == 200, legacy.text
    assert connected.agents().data[0].request_count == 259


def seed_spend(gateway: Gateway, source_id: str, vendor_id: str, email: str, cost: str, day: str) -> None:
    write_rows(
        'INSERT INTO "LiteLLM_ExternalCloudAgentSpend" '
        "(source_id,vendor_user_id,email,display_name,day,cost_micros,prompt_tokens,completion_tokens,requests) "
        "VALUES (%s,%s,%s,%s,%s::date,%s,40,10,2)",
        (source_id, vendor_id, email, vendor_id, day, int(Decimal(cost) * 1_000_000)),
        database_url=gateway.database_url,
    )


def test_spend_matches_email_rolls_up_teams_and_cascades_without_deleting_gateway_data(connected: Connected) -> None:
    gateway: Final = connected.gateway
    source_id: Final = connected.connection.source_id
    now: Final = datetime.now(timezone.utc)
    email: Final = f"{uuid.uuid4().hex}@example.test"
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        other_team: Final = scenario.team()
        user: Final = scenario.user(user_email=email, teams=[team])
        multi_email: Final = f"multi-{uuid.uuid4().hex}@example.test"
        scenario.user(user_email=multi_email, teams=[team, other_team])
        ambiguous_email: Final = f"ambiguous-{uuid.uuid4().hex}@example.test"
        duplicate_users: Final = (scenario.user(), scenario.user())
        for duplicate_user in duplicate_users:
            write_rows(
                'UPDATE "LiteLLM_UserTable" SET user_email=%s WHERE user_id=%s',
                (ambiguous_email, duplicate_user),
                database_url=gateway.database_url,
            )
        daily_id: Final = str(uuid.uuid4())
        write_rows(
            'INSERT INTO "LiteLLM_DailyUserSpend" (id,user_id,date,api_key,spend,updated_at) VALUES (%s,%s,%s,%s,4.25,NOW())',
            (daily_id, user, now.date().isoformat(), "external-cloud-agent-integration"),
            database_url=gateway.database_url,
        )
        try:
            seed_spend(gateway, source_id, "matched", email.upper(), "2.50", now.date().isoformat())
            seed_spend(gateway, source_id, "unknown", "unknown@example.test", "1.75", now.date().isoformat())
            seed_spend(gateway, source_id, "multi", multi_email, "0.000003", now.date().isoformat())
            seed_spend(gateway, source_id, "ambiguous", ambiguous_email.upper(), "0.75", now.date().isoformat())
            seed_spend(gateway, source_id, "out-of-window", email, "999", (now - timedelta(days=40)).date().isoformat())
            response: Final = gateway.request("GET", ROOT + "/spend", params={"source_id": source_id, "days": "30"})
            assert response.status_code == 200, response.text
            spend: Final = SpendReport.model_validate(response.json())
            matched: Final = next(row for row in spend.members if row.vendor_user_id == "matched")
            assert matched.litellm_user_id == user
            assert matched.litellm_gateway_spend_usd == Decimal("4.25")
            assert spend.totals.grok_cost_usd == Decimal("5.000003")
            assert spend.totals.litellm_gateway_spend_usd == Decimal("4.25")
            assert (spend.totals.tokens, spend.totals.requests) == (200, 8)
            rollups: Final = {row.team_name: row for row in spend.teams}
            assert next(row for row in spend.teams if row.team_id == team).grok_cost_usd == Decimal("2.50")
            assert rollups["Unassigned"].grok_cost_usd == Decimal("2.50")
            assert rollups["Multiple teams"].grok_cost_usd == Decimal("0.000003")
            assert {row.team_id for row in rollups["Multiple teams"].teams} == {team, other_team}
            assert next(row for row in spend.members if row.vendor_user_id == "ambiguous").litellm_user_id is None
            for rows in (spend.members, spend.teams):
                assert sum(row.grok_cost_usd for row in rows) == spend.totals.grok_cost_usd
                assert sum(row.litellm_gateway_spend_usd for row in rows) == spend.totals.litellm_gateway_spend_usd
                assert sum(row.tokens for row in rows) == spend.totals.tokens
                assert sum(row.requests for row in rows) == spend.totals.requests
            recorded: Final = connected.hook(frame())
            assert recorded.status_code == 200, recorded.text
            deleted: Final = gateway.request("DELETE", ROOT + f"/sources/{source_id}")
            assert deleted.status_code == 200, deleted.text
            assert gateway.rows('SELECT * FROM "LiteLLM_ExternalCloudAgent" WHERE source_id=%s', (source_id,)) == []
            assert (
                gateway.rows('SELECT * FROM "LiteLLM_ExternalCloudAgentSpend" WHERE source_id=%s', (source_id,)) == []
            )
            assert gateway.rows('SELECT id FROM "LiteLLM_DailyUserSpend" WHERE id=%s', (daily_id,)) == [
                {"id": daily_id}
            ]
        finally:
            write_rows(
                'DELETE FROM "LiteLLM_DailyUserSpend" WHERE id=%s', (daily_id,), database_url=gateway.database_url
            )
