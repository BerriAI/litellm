from datetime import datetime, timezone
from collections.abc import Sequence
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma import Prisma

from litellm.proxy.agent_endpoints.external_agents.governance import (
    ExternalAgentGovernance,
    MappedUser,
    decision_verdict,
    evaluate_policy,
    failure_decision,
    inferred_team,
)
from litellm.proxy.agent_endpoints.external_agents.models import (
    Agent,
    Decision,
    HookFailure,
    ObservedAgentEvent,
    Policy,
    Source,
    Verdict,
)

NOW: Final = datetime(2026, 11, 14, tzinfo=timezone.utc)
SOURCE: Final = Source(source_id="source", display_name="Bots", signing_secret="encrypted", mode="enforce")
EVENT: Final = ObservedAgentEvent(
    request_id="request",
    tenant_id="tenant",
    external_agent_id="bot",
    actor_id="actor",
    frame_type="tool_call",
    application="grok-bot",
    tool_names=("Shell",),
)
AGENT: Final = Agent(
    id="agent",
    source_id="source",
    external_agent_id="bot",
    team_id="team",
    owner_user_id="owner",
    first_seen_at=NOW,
    last_seen_at=NOW,
    request_count=1,
)


@pytest.mark.parametrize("teams,expected", [((), None), (("team",), "team"), (("one", "two"), None)])
def test_team_inference_requires_exactly_one_team(teams: tuple[str, ...], expected: str | None) -> None:
    assert inferred_team(MappedUser(user_id="user", teams=teams)) == expected


@pytest.mark.parametrize("tools", [("sHeLl",), ("server",)])
def test_policy_denies_tool_names_and_mcp_toolsets_case_insensitively(tools: tuple[str, ...]) -> None:
    event: Final = EVENT.model_copy(update={"tool_names": ("Shell", "CallMcpTool", "SERVER")})
    policy: Final = Policy(policy_id="policy", source_id="source", blocked_tools=tools)
    assert evaluate_policy(event, AGENT, None, policy) is not None


def test_prompt_does_not_apply_tool_blocklist() -> None:
    policy: Final = Policy(policy_id="policy", source_id="source", blocked_tools=("Shell",))
    assert evaluate_policy(EVENT.model_copy(update={"frame_type": "prompt"}), AGENT, None, policy) is None


def test_missing_policy_allows_unmapped_actor() -> None:
    assert evaluate_policy(EVENT, AGENT, None, None) is None


def test_blocked_bot_denied_without_policy() -> None:
    assert evaluate_policy(EVENT, AGENT.model_copy(update={"status": "blocked"}), None, None) == (
        "This bot is blocked by your administrator."
    )


def test_unmapped_actor_policy() -> None:
    policy: Final = Policy(policy_id="policy", source_id="source", allow_unmapped_actors=False)
    assert evaluate_policy(EVENT, AGENT, None, policy) == (
        "Your account has not been mapped to a LiteLLM user. Contact your administrator."
    )
    assert evaluate_policy(EVENT, AGENT, MappedUser(user_id="user"), policy) is None


def test_owner_required_policy() -> None:
    policy: Final = Policy(policy_id="policy", source_id="source", require_owner=True)
    assert evaluate_policy(EVENT, AGENT.model_copy(update={"owner_user_id": None}), None, policy) == (
        "This bot needs an owner. Contact your administrator."
    )
    assert evaluate_policy(EVENT, AGENT, None, policy) is None


@pytest.mark.parametrize("mode,expected", [("shadow", "allow"), ("enforce", "deny")])
def test_failure_verdict_obeys_mode_and_retains_would_deny(mode: str, expected: str) -> None:
    decision: Final = failure_decision(SOURCE.model_copy(update={"mode": mode}), EVENT, NOW)
    assert decision.verdict == expected
    assert decision.would_verdict == "deny"
    assert (decision_verdict(decision).deny_reason is None) == (mode == "shadow")
    assert "messages" not in decision.model_dump()


def database_with_rows(rows: Sequence[object]) -> MagicMock:
    db: Final = MagicMock(spec=Prisma)
    db.query_raw = AsyncMock(side_effect=rows)
    db.execute_raw = AsyncMock(return_value=1)
    db.tx.return_value.__aenter__ = AsyncMock(return_value=db)
    db.tx.return_value.__aexit__ = AsyncMock(return_value=False)
    return db


@pytest.mark.asyncio
async def test_retry_returns_stored_verdict_without_rediscovery() -> None:
    decision: Final = Decision(
        request_id="request",
        source_id="source",
        external_agent_id="bot",
        frame_type="tool_call",
        verdict="deny",
        would_verdict="deny",
        reason="Stored decision",
        enforced=True,
        created_at=NOW,
    )
    db: Final = database_with_rows(
        [[SOURCE.model_dump(exclude=set()) | {"signing_secret": "encrypted"}], [decision.model_dump()]]
    )
    result: Final = await ExternalAgentGovernance(db).evaluate(SOURCE, EVENT)
    assert result == Verdict(action="deny", deny_reason="Stored decision")
    assert db.query_raw.await_count == 2


@pytest.mark.asyncio
async def test_tenant_mismatch_rejected_before_discovery() -> None:
    db: Final = database_with_rows([[SOURCE.model_dump() | {"tenant_id": "other", "signing_secret": "encrypted"}]])
    result: Final = await ExternalAgentGovernance(db).evaluate(SOURCE, EVENT)
    assert result == HookFailure(status=401, reason="Tenant does not match source")
    assert db.execute_raw.await_count == 0


@pytest.mark.asyncio
async def test_config_test_does_not_discover_or_log() -> None:
    db: Final = database_with_rows([[SOURCE.model_dump() | {"signing_secret": "encrypted"}], [], []])
    result: Final = await ExternalAgentGovernance(db).evaluate(
        SOURCE,
        EVENT.model_copy(update={"application": "config-test"}),
    )
    assert result == Verdict(action="allow")
    assert db.execute_raw.await_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,action", [("shadow", "allow"), ("enforce", "deny")])
async def test_database_failure_returns_mode_safe_verdict(mode: str, action: str) -> None:
    db: Final = database_with_rows([])
    db.query_raw.side_effect = RuntimeError("database unavailable")
    result: Final = await ExternalAgentGovernance(db).evaluate(SOURCE.model_copy(update={"mode": mode}), EVENT)
    assert isinstance(result, Verdict)
    assert result.action == action
    assert db.execute_raw.await_count == 1


@pytest.mark.asyncio
async def test_shadow_logs_would_deny_but_returns_allow() -> None:
    source: Final = SOURCE.model_copy(update={"mode": "shadow", "tenant_id": "tenant"})
    policy: Final = Policy(policy_id="policy", source_id="source", blocked_tools=("Shell",))
    db: Final = database_with_rows(
        [[source.model_dump() | {"signing_secret": "encrypted"}], [], [], [AGENT.model_dump()], [policy.model_dump()]]
    )
    result: Final = await ExternalAgentGovernance(db).evaluate(source, EVENT)
    assert result == Verdict(action="allow")
    log_args: Final = db.execute_raw.call_args.args
    assert log_args[9:11] == ("allow", "deny")
    assert "messages" not in str(log_args)
