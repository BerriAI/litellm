import logging
import uuid
from datetime import datetime, timezone
from typing import Final

from anyio import fail_after
from pydantic import TypeAdapter

from litellm.proxy.agent_endpoints.external_agents.database import Database, Session
from litellm.proxy.agent_endpoints.external_agents.models import (
    Agent,
    Decision,
    FrozenModel,
    HookFailure,
    ObservedAgentEvent,
    Policy,
    Source,
    Verdict,
)

logger: Final = logging.getLogger("LiteLLM Proxy")


class MappedUser(FrozenModel):
    user_id: str
    teams: tuple[str, ...] = ()


class ExternalAgentStore:
    def __init__(self, db: Session) -> None:
        self.db = db

    async def source(self, source_id: str, lock: bool = False) -> Source | None:
        suffix: Final = " FOR UPDATE" if lock else ""
        rows: Final = TypeAdapter(list[Source]).validate_python(
            await self.db.query_raw('SELECT * FROM "ExternalAgentSource" WHERE source_id=$1' + suffix, source_id)
        )
        return rows[0] if rows else None

    async def decision(self, request_id: str) -> Decision | None:
        rows: Final = TypeAdapter(list[Decision]).validate_python(
            await self.db.query_raw(
                'SELECT * FROM "ExternalAgentDecision" WHERE request_id=$1',
                request_id,
            )
        )
        return rows[0] if rows else None

    async def actor(self, source_id: str, actor_id: str | None) -> MappedUser | None:
        rows: Final = TypeAdapter(list[MappedUser]).validate_python(
            await self.db.query_raw(
                'SELECT u.user_id, u.teams FROM "ExternalAgentActor" a JOIN "LiteLLM_UserTable" u '
                "ON u.user_id=a.user_id WHERE a.source_id=$1 AND a.external_actor_id=$2",
                source_id,
                actor_id,
            )
        )
        return rows[0] if rows else None

    async def discover(self, source_id: str, event: ObservedAgentEvent, user: MappedUser | None) -> Agent:
        team_id: Final = inferred_team(user)
        user_id: Final = user.user_id if user else None
        rows: Final = TypeAdapter(list[Agent]).validate_python(
            await self.db.query_raw(
                'INSERT INTO "ExternalAgent" (id, source_id, external_agent_id, team_id, owner_user_id, '
                "owner_inferred, request_count, last_actor_id, last_model) "
                "VALUES ($1,$2,$3,$4,$5,$6,1,$7,$8) ON CONFLICT (source_id,external_agent_id) DO UPDATE SET "
                'last_seen_at=NOW(), request_count="ExternalAgent".request_count+1, '
                "last_actor_id=EXCLUDED.last_actor_id, last_model=EXCLUDED.last_model, "
                'team_id=COALESCE("ExternalAgent".team_id,EXCLUDED.team_id), '
                'owner_user_id=COALESCE("ExternalAgent".owner_user_id,EXCLUDED.owner_user_id), '
                'owner_inferred=CASE WHEN "ExternalAgent".owner_user_id IS NULL AND EXCLUDED.owner_user_id IS NOT NULL '
                'THEN true ELSE "ExternalAgent".owner_inferred END RETURNING *',
                str(uuid.uuid4()),
                source_id,
                event.external_agent_id,
                team_id,
                user_id,
                user_id is not None,
                event.actor_id,
                event.model,
            )
        )
        return rows[0]

    async def policy(self, source: Source, team_id: str | None) -> Policy | None:
        rows: Final = TypeAdapter(list[Policy]).validate_python(
            await self.db.query_raw(
                'SELECT * FROM "ExternalAgentPolicy" WHERE source_id=$1 AND '
                "((team_id=$2 AND $2::text IS NOT NULL) OR policy_id=$3 OR team_id IS NULL) "
                "ORDER BY (team_id IS NOT NULL AND team_id=$2) DESC, (policy_id=$3) DESC NULLS LAST LIMIT 1",
                source.source_id,
                team_id,
                source.default_policy_id,
            )
        )
        return rows[0] if rows else None

    async def log(self, decision: Decision) -> None:
        await self.db.execute_raw(
            'INSERT INTO "ExternalAgentDecision" (request_id,source_id,external_agent_id,actor_id,user_id,team_id,'
            "frame_type,tool_names,verdict,would_verdict,reason,enforced,policy_id,created_at) "
            "VALUES ($1,$2,$3,$4,$5,$6,$7,$8::text[],$9,$10,$11,$12,$13,$14::timestamp) "
            "ON CONFLICT (request_id) DO NOTHING",
            decision.request_id,
            decision.source_id,
            decision.external_agent_id,
            decision.actor_id,
            decision.user_id,
            decision.team_id,
            decision.frame_type,
            list(decision.tool_names),
            decision.verdict,
            decision.would_verdict,
            decision.reason,
            decision.enforced,
            decision.policy_id,
            decision.created_at,
        )


def evaluate_policy(
    event: ObservedAgentEvent, agent: Agent, user: MappedUser | None, policy: Policy | None
) -> str | None:
    if agent.status == "blocked":
        return "This bot is blocked by your administrator."
    if policy is None:
        return None
    blocked: Final = frozenset(tool.casefold() for tool in policy.blocked_tools)
    denied: Final = next((tool for tool in event.tool_names if tool.casefold() in blocked), None)
    if event.frame_type == "tool_call" and denied:
        return f"Your team's policy does not allow the tool {denied}."[:500]
    if user is None and not policy.allow_unmapped_actors:
        return "Your account has not been mapped to a LiteLLM user. Contact your administrator."
    if policy.require_owner and agent.owner_user_id is None:
        return "This bot needs an owner. Contact your administrator."
    return None


def inferred_team(user: MappedUser | None) -> str | None:
    return user.teams[0] if user and len(user.teams) == 1 else None


def decision_verdict(decision: Decision) -> Verdict:
    return Verdict(action=decision.verdict, deny_reason=decision.reason if decision.verdict == "deny" else None)


def failure_decision(source: Source, event: ObservedAgentEvent, now: datetime) -> Decision:
    return Decision(
        request_id=event.request_id,
        source_id=source.source_id,
        external_agent_id=event.external_agent_id,
        actor_id=event.actor_id,
        frame_type=event.frame_type,
        tool_names=event.tool_names,
        verdict="deny" if source.mode == "enforce" else "allow",
        would_verdict="deny",
        reason="The governance check could not complete. Please retry or contact your administrator.",
        enforced=source.mode == "enforce",
        created_at=now,
    )


class ExternalAgentGovernance:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def evaluate(self, source: Source, event: ObservedAgentEvent) -> Verdict | HookFailure:
        try:
            with fail_after(3):
                return await self._evaluate(source, event)
        except Exception:
            decision: Final = failure_decision(source, event, datetime.now(timezone.utc))
            logger.exception("External agent governance evaluation failed for source %s", source.source_id)
            if event.application != "config-test":
                try:
                    with fail_after(0.3):
                        await ExternalAgentStore(self.db).log(decision)
                except Exception:
                    logger.exception("External agent failure decision could not be persisted")
            return decision_verdict(decision)

    async def _evaluate(self, source: Source, event: ObservedAgentEvent) -> Verdict | HookFailure:
        async with self.db.tx(timeout=2500) as transaction:
            store: Final = ExternalAgentStore(transaction)
            current: Final = await store.source(source.source_id, lock=True)
            if current is None:
                return HookFailure(status=404, reason="Unknown source")
            if current.tenant_id is not None and current.tenant_id != event.tenant_id:
                return HookFailure(status=401, reason="Tenant does not match source")
            if event.application == "config-test":
                return await self._config_test(store, current, event)
            await transaction.execute_raw("SELECT pg_advisory_xact_lock(hashtextextended($1,0))", event.request_id)
            previous: Final = await store.decision(event.request_id)
            if previous:
                if previous.source_id != current.source_id:
                    return HookFailure(status=401, reason="Webhook id belongs to another source")
                return decision_verdict(previous)
            if current.tenant_id is None:
                await transaction.execute_raw(
                    'UPDATE "ExternalAgentSource" SET tenant_id=$1,updated_at=NOW() WHERE source_id=$2',
                    event.tenant_id,
                    current.source_id,
                )
            user: Final = await store.actor(current.source_id, event.actor_id)
            agent: Final = await store.discover(current.source_id, event, user)
            policy: Final = await store.policy(current, agent.team_id)
            reason: Final = evaluate_policy(event, agent, user, policy)
            decision: Final = Decision(
                request_id=event.request_id,
                source_id=current.source_id,
                external_agent_id=event.external_agent_id,
                actor_id=event.actor_id,
                user_id=user.user_id if user else None,
                team_id=agent.team_id,
                frame_type=event.frame_type,
                tool_names=event.tool_names,
                verdict="deny" if reason and current.mode == "enforce" else "allow",
                would_verdict="deny" if reason else "allow",
                reason=reason,
                enforced=current.mode == "enforce",
                policy_id=policy.policy_id if policy else None,
                created_at=datetime.now(timezone.utc),
            )
            await store.log(decision)
            return decision_verdict(decision)

    async def _config_test(self, store: ExternalAgentStore, source: Source, event: ObservedAgentEvent) -> Verdict:
        user: Final = await store.actor(source.source_id, event.actor_id)
        policy: Final = await store.policy(source, inferred_team(user))
        now: Final = datetime.now(timezone.utc)
        agent: Final = Agent(
            id="config-test",
            source_id=source.source_id,
            external_agent_id=event.external_agent_id,
            team_id=inferred_team(user),
            owner_user_id=user.user_id if user else None,
            first_seen_at=now,
            last_seen_at=now,
            request_count=0,
        )
        reason: Final = evaluate_policy(event, agent, user, policy)
        return Verdict(
            action="deny" if reason and source.mode == "enforce" else "allow",
            deny_reason=reason if source.mode == "enforce" else None,
        )
