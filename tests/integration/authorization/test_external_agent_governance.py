import base64
import hashlib
import hmac
import json
import time
import uuid
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Final, Literal

import httpx
import pytest
from pydantic import JsonValue, TypeAdapter

from tests.integration._support.client import Gateway, object_value, string_value
from tests.integration._support.database import write_rows

OBJECTS: Final = TypeAdapter(tuple[Mapping[str, JsonValue], ...])


def test_zero_input_source_authentication_and_default_coverage(gateway: Gateway) -> None:
    response: Final = gateway.request("POST", "/v1/external_agents/sources")
    assert response.status_code == 200, response.text
    source: Final = object_value(response.json())
    identity: Final = string_value(source["source_id"])
    key: Final = string_value(source["ingest_key"])
    hook: Final = Hook(gateway, identity, "", key)
    try:
        assert source["display_name"] == "GrokBot" and source["mode"] == "shadow"
        assert key.startswith("litellm_ingest_")
        assert gateway.get(f"/v1/external_agents/sources/{identity}/connection")["ingest_key"] == key
        policies_response: Final = gateway.request(
            "GET", "/v1/external_agents/policies", params={"source_id": identity}
        )
        assert policies_response.status_code == 200, policies_response.text
        policies: Final = OBJECTS.validate_json(policies_response.content)
        assert len(policies) == 1
        assert (
            policies[0]["team_id"],
            policies[0]["blocked_tools"],
            policies[0]["allow_unmapped_actors"],
            policies[0]["require_owner"],
        ) == (None, [], True, False)
        assert hook.send("ignored", application="config-test").status_code == 200
        sources_response: Final = gateway.request("GET", "/v1/external_agents/sources")
        assert sources_response.status_code == 200, sources_response.text
        sources: Final = OBJECTS.validate_json(sources_response.content)
        saved: Final = next(item for item in sources if item["source_id"] == identity)
        assert saved["last_event_at"] is not None
        assert not {"ingest_key", "signing_secret", "admin_api_key"}.intersection(saved)
        assert hook.agents() == ()
        assert hook.send("bot", authorization="").status_code == 401
        assert hook.send("bot", authorization="Bearer wrong").status_code == 401
        assert hook.send("bot").status_code == 200
        summary: Final = gateway.get("/v1/external_agents/summary", {"source_id": identity})
        assert summary["covered_by_team_policy"] == summary["discovered"] == 1
        secret: Final = "whsec_" + base64.b64encode(b"test-hardening-key").decode()
        patched: Final = gateway.request("PATCH", f"/v1/external_agents/sources/{identity}", {"signing_secret": secret})
        assert patched.status_code == 200, patched.text
        assert hook.send("bad-signature").status_code == 401
        signed: Final = Hook(gateway, identity, secret, key).send("signed")
        assert signed.status_code == 200, signed.text
    finally:
        for table in ("Decision", "Actor", "Policy", ""):
            name: Final = "LiteLLM_ExternalAgent" + table
            write_rows(f'DELETE FROM "{name}" WHERE source_id=%s', (identity,))
        write_rows('DELETE FROM "LiteLLM_ExternalAgentSource" WHERE source_id=%s', (identity,))


@dataclass(frozen=True, slots=True)
class Hook:
    gateway: Gateway
    source_id: str
    secret: str = field(repr=False)
    ingest_key: str = field(default="", repr=False)

    def send(
        self,
        bot: str,
        actor: str = "member",
        *,
        request_id: str | None = None,
        tool: str | None = None,
        toolset: str | None = None,
        application: str = "grok-bot",
        tenant: str = "tenant",
        signature_valid: bool = True,
        authorization: str | None = None,
    ) -> httpx.Response:
        identity: Final = request_id or str(uuid.uuid4())
        body: Final = json.dumps(
            {
                "request_id": identity,
                "type": "tool_call" if tool else "prompt",
                "session_id": bot,
                "tenant_id": tenant,
                "actor": {"type": "user", "id": actor, "email_address": None},
                "source": {"application": application},
                "model": "external-model",
                "metadata": {},
                "messages": [
                    {
                        "role": "assistant" if tool else "user",
                        "content": [
                            {
                                "type": "tool_use",
                                "tool_name": tool,
                                "input": {"text": "PRIVATE-CONTENT-MARKER"},
                                "tool_info": {"type": "mcp", "toolset_name": toolset},
                            }
                            if tool
                            else {"type": "text", "text": "PRIVATE-CONTENT-MARKER"},
                        ],
                    }
                ],
            },
            separators=(",", ":"),
        ).encode()
        timestamp: Final = str(int(time.time()))
        signed: Final = f"{identity}.{timestamp}.".encode() + body
        digest: Final = hmac.new(base64.b64decode(self.secret[6:]), signed, hashlib.sha256).digest()
        return self.gateway.client.post(
            "/grok_bot/inference_hook/" + self.source_id,
            content=body,
            headers={
                "Content-Type": "application/json",
                "webhook-id": identity,
                "webhook-timestamp": timestamp,
                "webhook-signature": "v1," + (base64.b64encode(digest).decode() if signature_valid else "invalid"),
                "x-litellm-user-role": "proxy_admin",
                "Authorization": authorization if authorization is not None else "Bearer " + self.ingest_key,
            },
        )

    def agents(self, exceptions_only: bool = False) -> tuple[Mapping[str, JsonValue], ...]:
        response: Final = self.gateway.request(
            "GET",
            "/v1/external_agents",
            params={
                "source_id": self.source_id,
                "exceptions_only": str(exceptions_only).lower(),
            },
        )
        assert response.status_code == 200, response.text
        return OBJECTS.validate_json(response.content)

    def policy(self, team_id: str | None = None, blocked_tools: tuple[str, ...] = ()) -> None:
        response: Final = self.gateway.request(
            "PUT",
            "/v1/external_agents/policies",
            {
                "source_id": self.source_id,
                "team_id": team_id,
                "blocked_tools": list(blocked_tools),
            },
        )
        assert response.status_code == 200, response.text


@pytest.fixture
def hook(gateway: Gateway) -> Iterator[Hook]:
    secret: Final = "whsec_" + base64.b64encode(uuid.uuid4().bytes + uuid.uuid4().bytes).decode()
    source: Final = gateway.post(
        "/v1/external_agents/sources",
        {
            "display_name": "Integration hook",
            "signing_secret": secret,
            "mode": "enforce",
        },
    )
    source_id: Final = string_value(source["source_id"])
    try:
        yield Hook(gateway, source_id, secret, string_value(source["ingest_key"]))
    finally:
        write_rows('DELETE FROM "LiteLLM_ExternalAgentDecision" WHERE source_id=%s', (source_id,))
        write_rows('DELETE FROM "LiteLLM_ExternalAgentActor" WHERE source_id=%s', (source_id,))
        write_rows('DELETE FROM "LiteLLM_ExternalAgentPolicy" WHERE source_id=%s', (source_id,))
        write_rows('DELETE FROM "LiteLLM_ExternalAgent" WHERE source_id=%s', (source_id,))
        write_rows('DELETE FROM "LiteLLM_ExternalAgentSource" WHERE source_id=%s', (source_id,))
        assert (
            gateway.rows('SELECT source_id FROM "LiteLLM_ExternalAgentSource" WHERE source_id=%s', (source_id,)) == []
        )


def test_discovery_retry_team_inheritance_and_bulk_resolution(hook: Hook) -> None:
    with hook.gateway.scenario() as scenario:
        team: Final = scenario.team()
        other_team: Final = scenario.team()
        single: Final = scenario.user(teams=[team])
        multiple: Final = scenario.user(teams=[team, other_team])
        hook.gateway.post(
            "/v1/external_agents/actors/bulk",
            {
                "source_id": hook.source_id,
                "mappings": [
                    {"external_actor_id": "single", "user_id": single},
                    {"external_actor_id": "multiple", "user_id": multiple},
                ],
            },
        )
        request_id: Final = str(uuid.uuid4())
        first: Final = hook.send("shared-bot", "single", request_id=request_id)
        retry: Final = hook.send("shared-bot", "single", request_id=request_id)
        assert first.status_code == retry.status_code == 200, (first.text, retry.text)
        assert first.json() == retry.json() == {"action": "allow"}
        discovered: Final = hook.agents()[0]
        assert (
            discovered["team_id"],
            discovered["owner_user_id"],
            discovered["owner_inferred"],
            discovered["request_count"],
        ) == (team, single, True, 1)
        assert discovered["issues"] == ["owner inferred"]
        multi: Final = hook.send("multi-team-bot", "multiple")
        assert multi.status_code == 200 and multi.json() == {"action": "allow"}, multi.text
        multi_agent: Final = next(agent for agent in hook.agents() if agent["external_agent_id"] == "multi-team-bot")
        assert multi_agent["team_id"] is None
        assert multi_agent["owner_user_id"] == multiple
        assert multi_agent["issues"] == ["no team", "owner inferred"]
        hook.policy(team, ("Shell",))
        before: Final = hook.gateway.get("/v1/external_agents/summary", {"source_id": hook.source_id})
        assert {
            key: before[key]
            for key in ("discovered", "covered_by_team_policy", "needs_owner", "unmapped_actors", "denied_24h")
        } == {
            "discovered": 2,
            "covered_by_team_policy": 2,
            "needs_owner": 2,
            "unmapped_actors": 0,
            "denied_24h": 0,
        }
        assert len(hook.agents(exceptions_only=True)) == 2
        assigned: Final = hook.gateway.request(
            "PATCH",
            "/v1/external_agents/bulk",
            {
                "agent_ids": [agent["id"] for agent in hook.agents()],
                "team_id": other_team,
                "owner_user_id": multiple,
            },
        )
        assert assigned.status_code == 200 and assigned.json() == {"updated": 2}, assigned.text
        following: Final = hook.send("shared-bot", "single")
        assert following.status_code == 200, following.text
        preserved: Final = next(agent for agent in hook.agents() if agent["external_agent_id"] == "shared-bot")
        assert (preserved["team_id"], preserved["owner_user_id"], preserved["owner_inferred"]) == (
            other_team,
            multiple,
            False,
        )
        assert hook.agents(exceptions_only=True) == ()
        assert all(agent["issues"] == [] for agent in hook.agents())
        rows: Final = hook.gateway.rows(
            'SELECT to_jsonb(d) AS decision FROM "LiteLLM_ExternalAgentDecision" d WHERE source_id=%s AND request_id=%s',
            (hook.source_id, request_id),
        )
        assert len(rows) == 1
        assert "PRIVATE-CONTENT-MARKER" not in json.dumps(rows)


def test_config_test_is_not_discovered_and_tenant_or_signature_mismatch_is_rejected(hook: Hook) -> None:
    test: Final = hook.send("test-bot", application="config-test", tenant="config-tenant")
    assert test.status_code == 200 and test.json() == {"action": "allow"}, test.text
    assert hook.agents() == ()
    assert (
        hook.gateway.rows(
            'SELECT request_id FROM "LiteLLM_ExternalAgentDecision" WHERE source_id=%s', (hook.source_id,)
        )
        == []
    )
    real: Final = hook.send("real-bot")
    wrong_tenant: Final = hook.send("other-bot", tenant="other-tenant")
    forged: Final = hook.send("forged-bot", signature_valid=False)
    assert real.status_code == 200, real.text
    assert wrong_tenant.status_code == 401, wrong_tenant.text
    assert forged.status_code == 401, forged.text
    assert tuple(agent["external_agent_id"] for agent in hook.agents()) == ("real-bot",)


@pytest.mark.parametrize("mode,action", [("enforce", "deny"), ("shadow", "allow")])
def test_blocked_mcp_toolset_is_logged_and_shadow_mode_allows(
    hook: Hook, mode: Literal["enforce", "shadow"], action: str
) -> None:
    response: Final = hook.gateway.request("PATCH", "/v1/external_agents/sources/" + hook.source_id, {"mode": mode})
    assert response.status_code == 200, response.text
    hook.policy(blocked_tools=("restricted-server",))
    request_id: Final = str(uuid.uuid4())
    verdict: Final = hook.send("mcp-bot", tool="CallMcpTool", toolset="RESTRICTED-SERVER", request_id=request_id)
    assert verdict.status_code == 200 and verdict.json()["action"] == action, verdict.text
    decisions: Final = hook.gateway.rows(
        'SELECT verdict,would_verdict,enforced FROM "LiteLLM_ExternalAgentDecision" WHERE request_id=%s',
        (request_id,),
    )
    assert decisions == [{"verdict": action, "would_verdict": "deny", "enforced": mode == "enforce"}]
    summary: Final = hook.gateway.get("/v1/external_agents/summary", {"source_id": hook.source_id})
    assert (summary["discovered"], summary["denied_24h"], summary["unmapped_actors"]) == (
        1,
        1 if mode == "enforce" else 0,
        1,
    )
    assert len(hook.agents(exceptions_only=True)) == 1
    assert hook.agents()[0]["issues"] == ["no team", "no owner", "unmapped member", "denied recently"]


def test_summary_counts_distinct_unmapped_members_even_after_bot_invokers_change(hook: Hook) -> None:
    empty: Final = hook.gateway.get("/v1/external_agents/summary", {"source_id": hook.source_id})
    assert (empty["unmapped_actors"], empty["denied_24h"]) == (0, 0)
    assert hook.send("bot-one", actor="first-member").status_code == 200
    assert hook.send("bot-two", actor="first-member").status_code == 200
    shared: Final = hook.gateway.get("/v1/external_agents/summary", {"source_id": hook.source_id})
    assert (shared["discovered"], shared["unmapped_actors"]) == (2, 1)
    assert all("unmapped member" in agent["issues"] for agent in hook.agents(exceptions_only=True))
    assert hook.send("bot-one", actor="second-member").status_code == 200
    assert hook.send("bot-two", actor="second-member").status_code == 200
    distinct: Final = hook.gateway.get("/v1/external_agents/summary", {"source_id": hook.source_id})
    assert distinct["unmapped_actors"] == 2
    with hook.gateway.scenario() as scenario:
        user: Final = scenario.user()
        hook.gateway.post(
            "/v1/external_agents/actors/bulk",
            {"source_id": hook.source_id, "mappings": [{"external_actor_id": "second-member", "user_id": user}]},
        )
        mapped: Final = hook.gateway.get("/v1/external_agents/summary", {"source_id": hook.source_id})
        assert mapped["unmapped_actors"] == 1
        assert all("unmapped member" not in agent["issues"] for agent in hook.agents())


def test_secret_rotation_retains_previous_signatures_without_returning_secrets(hook: Hook) -> None:
    new_secret: Final = "whsec_" + base64.b64encode(uuid.uuid4().bytes).decode()
    response: Final = hook.gateway.request(
        "PATCH", "/v1/external_agents/sources/" + hook.source_id, {"signing_secret": new_secret}
    )
    assert response.status_code == 200, response.text
    serialized: Final = object_value(response.json())
    assert "signing_secret" not in serialized and "previous_signing_secret" not in serialized
    old: Final = hook.send("old-secret-bot")
    new: Final = Hook(hook.gateway, hook.source_id, new_secret, hook.ingest_key).send("new-secret-bot")
    assert old.status_code == new.status_code == 200, (old.text, new.text)
    assert old.json() == new.json() == {"action": "allow"}


def test_concurrent_retries_discover_and_count_once(hook: Hook) -> None:
    request_id: Final = str(uuid.uuid4())
    with ThreadPoolExecutor(max_workers=8) as executor:
        responses: Final = tuple(executor.map(lambda _: hook.send("shared-bot", request_id=request_id), range(8)))
    assert tuple(response.status_code for response in responses) == (200,) * 8
    assert tuple(response.json() for response in responses) == ({"action": "allow"},) * 8
    agents: Final = hook.agents()
    assert len(agents) == 1
    assert agents[0]["request_count"] == 1
    decisions: Final = hook.gateway.rows(
        'SELECT request_id FROM "LiteLLM_ExternalAgentDecision" WHERE source_id=%s',
        (hook.source_id,),
    )
    assert decisions == [{"request_id": request_id}]


def test_racing_retries_return_the_inserted_verdict_even_when_evaluations_differ(hook: Hook) -> None:
    hook.policy(blocked_tools=("Shell",))
    request_id: Final = str(uuid.uuid4())
    with ThreadPoolExecutor(max_workers=8) as executor:
        responses: Final = tuple(
            executor.map(
                lambda tool: hook.send("shared-bot", request_id=request_id, tool=tool),
                ("Shell", "ReadFile") * 4,
            )
        )
    assert tuple(response.status_code for response in responses) == (200,) * 8
    decisions: Final = hook.gateway.rows(
        'SELECT verdict,reason FROM "LiteLLM_ExternalAgentDecision" WHERE source_id=%s',
        (hook.source_id,),
    )
    assert len(decisions) == 1
    expected: Final = (
        {"action": "deny", "deny_reason": decisions[0]["reason"]}
        if decisions[0]["verdict"] == "deny"
        else {"action": "allow"}
    )
    assert tuple(response.json() for response in responses) == (expected,) * 8
    assert hook.agents()[0]["request_count"] == 1


def test_concurrent_first_frames_cannot_overwrite_the_bound_tenant(hook: Hook) -> None:
    with ThreadPoolExecutor(max_workers=2) as executor:
        responses: Final = tuple(
            executor.map(lambda tenant: hook.send(tenant, tenant=tenant), ("tenant-one", "tenant-two"))
        )
    assert sorted(response.status_code for response in responses) == [200, 401]
    agents: Final = hook.agents()
    assert len(agents) == 1
    source: Final = hook.gateway.rows(
        'SELECT tenant_id FROM "LiteLLM_ExternalAgentSource" WHERE source_id=%s',
        (hook.source_id,),
    )
    assert source == [{"tenant_id": agents[0]["external_agent_id"]}]


@pytest.mark.parametrize(
    ("email", "alias", "label"),
    (
        ("member@example.test", "Member alias", "member@example.test"),
        (None, "Member alias", "Member alias"),
        (None, None, None),
    ),
)
def test_management_displays_current_user_and_bot_labels(
    hook: Hook, email: str | None, alias: str | None, label: str | None
) -> None:
    with hook.gateway.scenario() as scenario:
        user: Final = scenario.user(user_email=email, user_alias=alias)
        hook.gateway.post(
            "/v1/external_agents/actors/bulk",
            {"source_id": hook.source_id, "mappings": [{"external_actor_id": "member", "user_id": user}]},
        )
        assert hook.send("bot-id").status_code == 200
        agent: Final = hook.agents()[0]
        assert agent["owner_display_name"] == label
        patched: Final = hook.gateway.request(
            "PATCH", "/v1/external_agents/bulk", {"agent_ids": [agent["id"]], "display_name": "Friendly bot"}
        )
        assert patched.status_code == 200, patched.text
        response: Final = hook.gateway.request(
            "GET", "/v1/external_agents/decisions", params={"source_id": hook.source_id}
        )
        assert response.status_code == 200, response.text
        decisions: Final = OBJECTS.validate_json(response.content)
        assert (decisions[0]["user_display_name"], decisions[0]["user_id"], decisions[0]["agent_display_name"]) == (
            label,
            user,
            "Friendly bot",
        )
