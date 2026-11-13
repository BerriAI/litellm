import logging
import uuid
from datetime import datetime, timezone
from typing import Final

from anyio import fail_after
from pydantic import TypeAdapter

from litellm.proxy.agent_endpoints.external_agents.cache import GovernanceMetadata
from litellm.proxy.agent_endpoints.external_agents.database import Session
from litellm.proxy.agent_endpoints.external_agents.models import (
    Agent,
    Decision,
    HookFailure,
    MappedUser,
    ObservedAgentEvent,
    Policy,
    Source,
    Verdict,
)

logger: Final = logging.getLogger("LiteLLM Proxy")
SOURCE_ROWS: Final = TypeAdapter(tuple[Source, ...])
USER_ROWS: Final = TypeAdapter(tuple[MappedUser, ...])
AGENT_ROWS: Final = TypeAdapter(tuple[Agent, ...])
POLICY_ROWS: Final = TypeAdapter(tuple[Policy, ...])
DECISION_ROWS: Final = TypeAdapter(tuple[Decision, ...])


class ExternalAgentStore:
    def __init__(self, db: Session, cache: GovernanceMetadata | None = None) -> None:
        self.db = db
        self.cache = cache

    async def source(self, source_id: str) -> Source | None:
        if self.cache is not None:
            return await self.cache.sources.get(source_id, lambda: self._source(source_id))
        return await self._source(source_id)

    async def _source(self, source_id: str) -> Source | None:
        rows: Final = SOURCE_ROWS.validate_python(
            await self.db.query_raw('SELECT * FROM "LiteLLM_ExternalAgentSource" WHERE source_id=$1', source_id)
        )
        return rows[0] if rows else None

    async def bind_tenant(self, source_id: str, tenant_id: str) -> HookFailure | None:
        if self.cache is not None:
            return await self.cache.bindings.get(
                (source_id, tenant_id), lambda: self._bind_tenant(source_id, tenant_id)
            )
        return await self._bind_tenant(source_id, tenant_id)

    async def _bind_tenant(self, source_id: str, tenant_id: str) -> HookFailure | None:
        await self.db.execute_raw(
            'UPDATE "LiteLLM_ExternalAgentSource" SET tenant_id=$2,updated_at=NOW() '
            "WHERE source_id=$1 AND tenant_id IS NULL",
            source_id,
            tenant_id,
        )
        if self.cache is not None:
            self.cache.sources.invalidate()
        bound: Final = await self.source(source_id)
        if bound is None:
            return HookFailure(status=404, reason="Unknown source")
        if bound.tenant_id != tenant_id:
            return HookFailure(status=401, reason="Tenant does not match source")
        return None

    async def decision(self, request_id: str) -> Decision | None:
        rows: Final = DECISION_ROWS.validate_python(
            await self.db.query_raw(
                'SELECT * FROM "LiteLLM_ExternalAgentDecision" WHERE request_id=$1',
                request_id,
            )
        )
        return rows[0] if rows else None

    async def actor(self, source_id: str, actor_id: str | None) -> MappedUser | None:
        if actor_id is None:
            return None
        if self.cache is not None:
            return await self.cache.actors.get((source_id, actor_id), lambda: self._actor(source_id, actor_id))
        return await self._actor(source_id, actor_id)

    async def _actor(self, source_id: str, actor_id: str) -> MappedUser | None:
        rows: Final = USER_ROWS.validate_python(
            await self.db.query_raw(
                'SELECT u.user_id, u.teams FROM "LiteLLM_ExternalAgentActor" a JOIN "LiteLLM_UserTable" u '
                "ON u.user_id=a.user_id WHERE a.source_id=$1 AND a.external_actor_id=$2",
                source_id,
                actor_id,
            )
        )
        return rows[0] if rows else None

    async def discover(self, source_id: str, event: ObservedAgentEvent, user: MappedUser | None) -> Agent:
        team_id: Final = inferred_team(user)
        user_id: Final = user.user_id if user else None
        rows: Final = AGENT_ROWS.validate_python(
            await self.db.query_raw(
                'INSERT INTO "LiteLLM_ExternalAgent" (id, source_id, external_agent_id, team_id, owner_user_id, '
                "owner_inferred, request_count, last_actor_id, last_model) "
                "VALUES ($1,$2,$3,$4,$5,$6,0,$7,$8) ON CONFLICT (source_id,external_agent_id) DO UPDATE SET "
                "last_seen_at=NOW(), "
                "last_actor_id=EXCLUDED.last_actor_id, last_model=EXCLUDED.last_model, "
                'team_id=COALESCE("LiteLLM_ExternalAgent".team_id,EXCLUDED.team_id), '
                'owner_user_id=COALESCE("LiteLLM_ExternalAgent".owner_user_id,EXCLUDED.owner_user_id), '
                'owner_inferred=CASE WHEN "LiteLLM_ExternalAgent".owner_user_id IS NULL AND EXCLUDED.owner_user_id IS NOT NULL '
                'THEN true ELSE "LiteLLM_ExternalAgent".owner_inferred END RETURNING *',
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
        if self.cache is not None:
            return await self.cache.policies.get(
                (source.source_id, team_id, source.default_policy_id), lambda: self._policy(source, team_id)
            )
        return await self._policy(source, team_id)

    async def _policy(self, source: Source, team_id: str | None) -> Policy | None:
        rows: Final = POLICY_ROWS.validate_python(
            await self.db.query_raw(
                'SELECT * FROM "LiteLLM_ExternalAgentPolicy" WHERE source_id=$1 AND '
                "((team_id=$2 AND $2::text IS NOT NULL) OR policy_id=$3 OR team_id IS NULL) "
                "ORDER BY (team_id IS NOT NULL AND team_id=$2) DESC, (policy_id=$3) DESC NULLS LAST LIMIT 1",
                source.source_id,
                team_id,
                source.default_policy_id,
            )
        )
        return rows[0] if rows else None

    async def log(self, decision: Decision) -> Decision | None:
        rows: Final = DECISION_ROWS.validate_python(
            await self.db.query_raw(
                'WITH inserted AS (INSERT INTO "LiteLLM_ExternalAgentDecision" (request_id,source_id,external_agent_id,actor_id,user_id,team_id,'
                "frame_type,tool_names,verdict,would_verdict,reason,enforced,policy_id,created_at) "
                "VALUES ($1,$2,$3,$4,$5,$6,$7,$8::text[],$9,$10,$11,$12,$13,$14::timestamp) "
                "ON CONFLICT (request_id) DO NOTHING RETURNING *), counted AS ("
                'UPDATE "LiteLLM_ExternalAgent" a SET request_count=a.request_count+1 FROM inserted d '
                "WHERE a.source_id=d.source_id AND a.external_agent_id=d.external_agent_id) SELECT * FROM inserted",
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
        )
        return rows[0] if rows else await self.decision(decision.request_id)


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


def stored_verdict(source: Source, decision: Decision) -> Verdict | HookFailure:
    if decision.source_id != source.source_id:
        return HookFailure(status=401, reason="Webhook id belongs to another source")
    return decision_verdict(decision)


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
    def __init__(self, db: Session, cache: GovernanceMetadata | None = None) -> None:
        self.store = ExternalAgentStore(db, cache)

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
                        stored: Final = await self.store.log(decision)
                        if stored is not None:
                            return stored_verdict(source, stored)
                except Exception:
                    logger.exception("External agent failure decision could not be persisted")
            return decision_verdict(decision)

    async def _evaluate(self, source: Source, event: ObservedAgentEvent) -> Verdict | HookFailure:
        if source.tenant_id is not None and source.tenant_id != event.tenant_id:
            return HookFailure(status=401, reason="Tenant does not match source")
        if event.application == "config-test":
            return await self._config_test(self.store, source, event)
        if source.tenant_id is None:
            binding_failure: Final = await self.store.bind_tenant(source.source_id, event.tenant_id)
            if binding_failure:
                return binding_failure
        user: Final = await self.store.actor(source.source_id, event.actor_id)
        agent: Final = await self.store.discover(source.source_id, event, user)
        policy: Final = await self.store.policy(source, agent.team_id)
        reason: Final = evaluate_policy(event, agent, user, policy)
        decision: Final = Decision(
            request_id=event.request_id,
            source_id=source.source_id,
            external_agent_id=event.external_agent_id,
            actor_id=event.actor_id,
            user_id=user.user_id if user else None,
            team_id=agent.team_id,
            frame_type=event.frame_type,
            tool_names=event.tool_names,
            verdict="deny" if reason and source.mode == "enforce" else "allow",
            would_verdict="deny" if reason else "allow",
            reason=reason,
            enforced=source.mode == "enforce",
            policy_id=policy.policy_id if policy else None,
            created_at=datetime.now(timezone.utc),
        )
        stored: Final = await self.store.log(decision)
        return (
            stored_verdict(source, stored)
            if stored
            else decision_verdict(failure_decision(source, event, datetime.now(timezone.utc)))
        )

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
